import socket

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 9999))

    try:
        while True:
            message = input("메시지를 입력하세요: ")
            if message.lower() == 'quit':
                break
            client_socket.send(message.encode('utf-8'))
            response = client_socket.recv(4096).decode('utf-8')
            print(f"{response}")
    except KeyboardInterrupt:
        print("클라이언트가 종료되었습니다.")
    finally:
        client_socket.close()

if __name__ == '__main__':
    start_client()
