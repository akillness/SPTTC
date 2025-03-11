
import asyncio
import websockets
import time
from datetime import datetime
import torch.multiprocessing as mp
import torch

import json

from deepseek_r1 import deepseek_r1, device

def worker_process(model):
    """웹소켓 요청을 처리하는 워커 프로세스"""
    # 프로세스별 모델 로드
    # model = deepseek_r1(
    #     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #     cached_dir='./',
    #     device=device
    # )
    print('Worker is available state')
    
    async def handler(websocket):
        print(f"Connected client : {websocket.remote_address}")
        try:
            async for message in websocket:
                # print(f"Waiting Message..")
                messagetojson = json.loads(message)
                
                if 'content' not in messagetojson:
                    # print("Invalid message format")
                    continue

                print(f"[Query] : {messagetojson['content']}")
                
                # 스트리밍 생성기 시작
                streamer = await asyncio.get_event_loop().run_in_executor(
                    None,
                    model.generate_stream,
                    messagetojson['content']
                )

                # 실시간 스트리밍 처리
                full_response = ""
                while True:
                    try:
                        token = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: next(streamer, None)
                        )
                        if token is None:
                            break
                            
                        full_response += token
                        
                        # 실시간 업데이트 메시지 전송
                        messagetojson.update({
                            'content': full_response,
                            'isMe': False,
                            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                            'isStreaming': True,  # 스트리밍 진행 중 여부
                        })

                        await websocket.send(json.dumps(messagetojson))
                        
                    except Exception as e:
                        print(f"스트리밍 오류: {e}")
                        break
                
                # messagetojson['isStreaming'] = False
                print(f"[전체 응답 완료]\n{full_response}")

        except websockets.exceptions.ConnectionClosedOK:
            print(f"클라이언트 연결 종료: {websocket.remote_address}")
        except Exception as e:
            print(f"에러 발생: {e}")
        finally:
            print(f"연결 종료 처리 완료: {websocket.remote_address}")

    # return handler
    '''
    async def handler(websocket):
        # 클라이언트의 웹소켓 연결을 처리하는 핸들러
        print(f"Connected client : {websocket.remote_address}")
        try:
            async for message in websocket:
                print(f"Waiting Message..")
                messagetojson = json.loads(message)
                 # 'content' 키가 존재하는지 확인
                content = messagetojson.get('content')
                if content is None:
                    print("Received message without 'content' key.")
                    continue  # 'content' 키가 없으면 다음 메시지로 넘어감
                
                # 비동기 실행을 위해 이벤트 루프에서 실행
                loop = asyncio.get_event_loop()
                print(messagetojson)
                print(f"[Query] : {messagetojson['content']}")
                response = await loop.run_in_executor(
                    None,  # 기본 Executor 사용 (ThreadPool)
                    model.generate,  # 모델 추론 메서드
                    messagetojson['content']       # 입력 메시지
                )
                
                print(f"[Server] response: {response}")
                
                # Update Message 
                messagetojson['content'] = response
                messagetojson['isMe'] = False
                messagetojson['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            
                result = json.dumps(messagetojson)
                await websocket.send(result)
                
        except websockets.exceptions.ConnectionClosedOK:
            print(f"클라이언트 {websocket.remote_address} 정상적으로 연결 종료")
        except Exception as e:
            print(f"웹소켓 에러: {e}")
        finally:
            print(f"클라이언트 {websocket.remote_address} 연결 해제됨")
    '''
    async def main():
        # SO_REUSEADDR 및 reuse_port 설정으로 다중 바인딩 허용
        async with websockets.serve(
            handler,
            # "0.0.0.0",          
            "localhost",  
            8080,
            reuse_port=True,
            ping_interval=None
        ):
            await asyncio.Future()  # 무한 실행

    asyncio.run(main())

def start_server():
    """서버 시작 함수"""
    # 메인 프로세스에서 워커 수 결정을 위한 임시 모델 로드
    model = deepseek_r1(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        cached_dir='./',
        device=device
    )
    num_workers = model.get_usable_process_num()
    # model.cleanup()  # 임시 모델 메모리 해제

    workers = []
    
    try:
        # 워커 프로세스 생성
        for _ in range(num_workers):
            p = mp.Process(target=worker_process,args={model})
            p.start()
            workers.append(p)
        
        print(f"🚀 WebSocket server started on ws://localhost:8080 (Workers: {num_workers})")
        print("🔌 Ctrl+C to stop the server")
        
        # 워커 모니터링 루프
        while True:
            for p in workers:
                if not p.is_alive():
                    print(f"⚠️ Worker {p.pid} died, restarting...")
                    p.join()
                    new_p = mp.Process(target=worker_process)
                    new_p.start()
                    workers.remove(p)
                    workers.append(new_p)
            # time.sleep(1)  # CPU 부하 조절

    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    finally:
        # 자원 정리
        for p in workers:
            if p.is_alive():
                p.terminate()
        model.cleanup()
        print("✅ Server shutdown completed")

if __name__ == '__main__':
    # 시작 방법 설정 (최초 1회만)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
        
    try:
        start_server()
    except Exception as e:
        print(f"❌ Critical server error: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Cleaned up GPU resources")