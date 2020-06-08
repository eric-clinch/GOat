import socket
import pickle
import io
import sys


# Given a nonnegative integer x, returns a bytes encoding.
# The inverse of this is BytesToSize
def SizeToBytes(x: int) -> bytes:
    assert(x >= 0)
    result = b''
    while x > 0:
        least_sig = x % 256
        result += bytes([least_sig])
        x //= 256
    return result


# The inverse of SizeToBytes
def BytesToSize(b: bytes) -> int:
    result = 0
    mult = 1
    for char in b:
        result += mult * char
        mult *= 256
    return result


# Produces a socket meant for the host of the distributed system
def ServerSocket(address: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((address, port))
    server.listen()
    return server


# Produces a socket mean for a worker in the distributed system
def WorkerSocket(address: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((address, port))
    return server


def Send(sckt: socket.socket, msg: bytes):
    msg_len_bytes: bytes = SizeToBytes(len(msg))

    # Always make the length portion 4 characters long
    assert(len(msg_len_bytes) <= 4)
    while len(msg_len_bytes) < 4:
        msg_len_bytes += b'\x00'

    # Prepend the message with its length
    sckt.send(msg_len_bytes + msg)

def Receive(sckt: socket.socket) -> bytes:
    msg: bytes = sckt.recv(4096)
    msg_len_bytes, msg = msg[:4], msg[4:]
    msg_len: int = BytesToSize(msg_len_bytes)
    while len(msg) < msg_len:
        msg += sckt.recv(4096)
    assert(msg_len == len(msg))
    return msg
