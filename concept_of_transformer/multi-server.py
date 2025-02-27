import socket
import multiprocessing
import torch
from deepseek_r1 import deepseek_r1, device



def handle_client(client_socket, client_address, deepbot):
    """클라이언트 요청 처리 핸들러"""
    print(f"Connected client : {client_address}")
    
    try:
        while True:
            message = client_socket.recv(4096).decode('utf-8')
            if not message:
                break
            print(f"[Query] {client_address}: {message}...")
            result = deepbot.generate(message)
            response = f"Server response: {result}"
            client_socket.send(response.encode('utf-8'))
    except Exception as e:
        print(f"Handle error: {e}")
    finally:
        client_socket.close()
        print(f"Disconneted: {client_address}")

def worker_process(server_socket, deepbot):
    """워커 프로세스 메인 로직"""
    try:
        while True:
            client_sock, addr = server_socket.accept()
            handle_client(client_sock, addr, deepbot)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        # 자원 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def start_server():
    """서버 시작 함수"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 9999))
    server.listen()

    # 모델 초기화 (한 번만 로드하여 모든 프로세스에서 공유)
    model = deepseek_r1(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        cached_dir='./',
        device=device
    )

    # CPU 코어 수 기반 워커 풀 생성
    # num_workers = multiprocessing.cpu_count()   # I/O 바운드 작업 가정
    # 모델의 메모리 사용량 (GB)

    num_workers = model.get_usable_process_num()    
    workers = []
    
    try:
        # 워커 프로세스 생성
        for _ in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(server, model)  # server와 model을 각각 넘겨준다.
            )
            p.start()
            workers.append(p)
        
        print(f"Starting server (Worker num: {num_workers})")
        while True:
            # 메인 프로세스는 관리자 역할만 수행
            for p in workers:
                if not p.is_alive():
                    print("Restarting workers...")
                    p.join()
                    new_p = multiprocessing.Process(
                        target=worker_process,
                        args=(server, model)
                    )
                    new_p.start()
                    workers.remove(p)
                    workers.append(new_p)
            multiprocessing.active_children()  # 좀비 프로세스 정리
            
    except KeyboardInterrupt:
        print("\nAccepted signal to quit")
    finally:
        # 자원 정리
        for p in workers:
            if p.is_alive():
                p.terminate()
        server.close()
        print("Quit server completed")

if __name__ == '__main__':
    try:
        start_server()
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleaned server system")
