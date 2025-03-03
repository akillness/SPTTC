import socket
import multiprocessing

def receive_response(client_socket):
    """서버로부터의 응답을 수신하여 출력하는 함수"""
    while True:
        try:
            response = client_socket.recv(4096).decode('utf-8')
            if not response:
                break
            print(f"서버 응답: {response}")
        except Exception as e:
            print(f"응답 수신 오류: {e}")
            break

def start_client():
    """클라이언트 시작 함수"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 8080))

    # 서버로부터의 응답을 수신하는 프로세스 시작
    receive_process = multiprocessing.Process(target=receive_response, args=(client_socket,))
    receive_process.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
    receive_process.start()

    try:
        while True:
            message = input("메시지를 입력하세요: ")
            if message.lower() == 'quit':
                break
            client_socket.send(message.encode('utf-8'))
    except KeyboardInterrupt:
        print("클라이언트가 종료되었습니다.")
    finally:
        client_socket.close()

if __name__ == '__main__':
    start_client()
