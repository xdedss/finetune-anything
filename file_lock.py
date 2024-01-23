
import os
import time


class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None

    def __enter__(self):
        spin_chars = ['|', '/', '-', '\\']
        spin_i = 0
        while True:
            try:
                self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                print(f"We have the lock {self.lock_file}          ")
                return self
            except FileExistsError:
                spin_i = (spin_i + 1) % len(spin_chars)
                print(f"Waiting for file lock {spin_chars[spin_i]}", end='\r')
                time.sleep(0.2)

    def __exit__(self, exc_type, exc_value, traceback):
        os.close(self.lock_fd)
        os.unlink(self.lock_file)
        print(f"We have released the lock {self.lock_file}")


if __name__ == '__main__':
    # test it
    with FileLock('lock.lock'):
        for i in range(20):
            print(i)
            time.sleep(1)
