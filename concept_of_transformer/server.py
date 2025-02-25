import socket
import multiprocessing

import torch
from deepseek_r1 import deepseek_r1, device

def handle_client(client_socket, client_address, deepbot):
    print(f"클라이언트 {client_address}와 연결되었습니다.")
    
    try:
        while True:
            message = client_socket.recv(4096).decode('utf-8')
            if not message:
                break
            print(f"클라이언트 {client_address}로부터: {message}")
            result = deepbot.generate(message)
            client_socket.send(f"서버로부터: {result}".encode('utf-8'))
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        client_socket.close()
        print(f"클라이언트 {client_address}와의 연결이 종료되었습니다.")

def start_server(bot):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9999))
    server.listen()
    print("서버가 시작되었습니다. 클라이언트의 연결을 기다립니다...")

    try:
        while True:
            client_socket, client_address = server.accept()
            process = multiprocessing.Process(target=handle_client, args=(client_socket, client_address, bot))
            process.start()
    except KeyboardInterrupt:
        print("\n서버 종료 요청 수신")
    finally:
        server.close()
        print("서버 소켓이 닫혔습니다")

if __name__ == '__main__':
    # 모델 이름
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    cached_dir = './'
    deepbot = None
    
    try:
        deepbot = deepseek_r1(model_name=model_name, cached_dir=cached_dir, device=device)
        start_server(deepbot)
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if deepbot is not None:
            deepbot.cleanup()
        # 추가적인 CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("모든 리소스 정리 완료")
