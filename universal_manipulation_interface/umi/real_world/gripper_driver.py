import serial
import time
import threading
from threading import Lock

OPEN = 0.083
CLOSE = 0.012

class GripperDriver:
    def __init__(self, serial_name='/dev/ttyUSB0', baudrate=115200, launch_timeout=3):
        self.serial_name = serial_name
        self.baudrate = baudrate
        self.launch_timeout = launch_timeout
        self.serial = serial.Serial(self.serial_name, self.baudrate, timeout=self.launch_timeout)
        time.sleep(2)  # Wait for the serial connection to initialize
    
    def is_serial_ready(self):
        return self.serial.is_open

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.serial.is_open:
            self.serial.close()
        print("Serial connection closed.")

    def send_target_width(self, target_width):
        try:
            new_target_width = float(target_width)
            # Ensure target_width is within the expected range (CLOSE-OPEN)
            if new_target_width > OPEN: new_target_width = OPEN
            if new_target_width < CLOSE: new_target_width = CLOSE
            self.serial.write(f"{new_target_width}\n".encode())
            time.sleep(0.005)  # Give the Arduino time to process the command
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    def read_sensor_width(self):
        self.serial.write("Width\n".encode('utf-8'))
        time.sleep(0.005)  # Wait for serial data to be available
        if self.serial.in_waiting > 0:
            raw_data = self.serial.readline()
            try:
                width_str = raw_data.decode('utf-8').strip()
                width = float(width_str)
                return width
            except UnicodeDecodeError:
                print("Received data could not be decoded. Raw data:", raw_data)
            except ValueError:
                print("Received data is not a valid float. Data:", raw_data)
        return None


# For testing
if __name__ == "__main__":
    try:
        with GripperDriver('/dev/ttyUSB0', 115200) as gripper:
            while True:
                user_input = input("Enter new target width (0.012 to 0.083) or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
                gripper.send_target_width(user_input)

    except KeyboardInterrupt:
        print("Exiting...")
