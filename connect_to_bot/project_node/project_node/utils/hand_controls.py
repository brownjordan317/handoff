"""
This code defines a class `HandControls` that processes hand positions to 
generate drive commands for a robot. The `call_controls` method takes the 
current positions of the left and right hands as input and generates a drive 
command based on the positions. The drive commands are "FORWARD", "LEFT", 
"RIGHT", and "STOP". Additionally, if both hands are in the "L" position, 
the drive command is set to "Unlock".
"""

class HandControls():
    def __init__(self):
        self.patience = 3
        self.current_call = None
        self.prev_call = None
        self.call_counter = 0
        self.drive_command = None

        pass

    def call_controls(self, left_hand, right_hand):
        """
        Determine the drive command based on the current hand positions.

        Args:
            left_hand (str): The current position of the left hand.
            right_hand (str): The current position of the right hand.

        Returns:
            str: The drive command to be sent to the robot.
        """

        # print(f"Left hand: {left_hand}, Right hand: {right_hand}")

        if left_hand == "V" and \
            right_hand == "V":
            
            if self.prev_call != "VV":
                self.call_counter = 1
                self.prev_call = "VV"
            else:
                self.call_counter += 1

            if self.call_counter >= self.patience:
                # print("Drive command: FORWARD")
                self.drive_command = "FORWARD"
            
        elif left_hand == "V" and \
            right_hand != "V":
            
            if self.prev_call != "VNone":
                self.call_counter = 1
                self.prev_call = "VNone"
            else:
                self.call_counter += 1

            if self.call_counter >= self.patience:
                # print("Drive command: LEFT")
                self.drive_command = "LEFT"
            
        elif left_hand != "V" and \
            right_hand == "V":
            
            if self.prev_call != "NoneV":
                self.call_counter = 1
                self.prev_call = "NoneV"
            else:
                self.call_counter += 1

            if self.call_counter >= self.patience:
                # print("Drive command: RIGHT")
                self.drive_command = "RIGHT"

        else:
            if self.prev_call != "NoneNone":
                self.call_counter = 1
                self.prev_call = "NoneNone"
            else:
                self.call_counter += 1
            # print("Drive command: STOP")
            self.drive_command = "STOP"

        if left_hand =="L" and right_hand == "L":
            self.drive_command = "Unlock"
