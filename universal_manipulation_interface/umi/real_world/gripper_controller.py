import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.real_world.gripper_driver import GripperDriver
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

OPEN = 0.083
CLOSE = 0.012
BIAS = 0.01
class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class GripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            serial_name,
            baudrate=115200,
            frequency=20,
            move_max_speed=200.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            verbose=False
            ):
        super().__init__(name="GripperController")
        self.serial_name = serial_name
        self.baudrate = baudrate
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.gripper_driver = GripperDriver(serial_name=self.serial_name, baudrate=self.baudrate, launch_timeout = self.launch_timeout)

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[GripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return True
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        try:
            # Home gripper to initialize (uncomment if needed)
            self.gripper_driver.send_target_width(OPEN)
            # main loop
            dt = 1. / self.frequency
            
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            target_pos_test = 0.0
            while keep_running:
                if self.gripper_driver.is_serial_ready() is False:
                    print(f"[GripperController] Serial connection is not ready: {self.serial_name}")
                    continue

                # update the state
                curr_position = self.gripper_driver.read_sensor_width()
                state = {
                    'gripper_state': 0,
                    'gripper_position': curr_position,
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)
                

                # Fetch command from queue
                try:
                    commands = self.input_queue.get_last_k()
                    # print("INPUT QUEUE GRIPPER:",commands)
                    n_cmd = len(commands['cmd'])
                    # print("LEN QUEUE GRIPPER:",n_cmd)
                except Empty:
                    n_cmd = 0
                
                # Execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        if (target_pos < 0.07):
                            self.gripper_driver.send_target_width(target_pos-BIAS)
                            # print("TARGET:POSE", target_pos-BIAS)
                        else:
                            self.gripper_driver.send_target_width(target_pos)
                        
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[GripperController] Disconnected from serial: {self.serial_name}")

