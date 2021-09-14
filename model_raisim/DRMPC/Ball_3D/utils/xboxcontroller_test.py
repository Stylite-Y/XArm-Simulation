from xbox360controller import Xbox360Controller
import time
import signal

xbox = Xbox360Controller(0, axis_threshold=0.02)
buttons = xbox.buttons
axes = xbox.axes
while True:
    time.sleep(0.1)
    for button in buttons:
        print(f'{button.name}: {button.is_pressed}')
    for axis in axes:
        if 'trigger' in axis.name:
            print(f'{axis.name}: {axis.value}')
        else:
            print(f'{axis.name}: {axis.x}, {axis.y}')
            print(xbox.hat.when_moved)

# def on_button_pressed(button):
#     print('Button {0} was pressed'.format(button.name))


# def on_button_released(button):
#     print('Button {0} was released'.format(button.name))


# def on_axis_moved(axis):
#     print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))

# try:
#     with Xbox360Controller(0, axis_threshold=0.2) as controller:
#         # Button A events
#         controller.button_a.when_pressed = on_button_pressed
#         controller.button_a.when_released = on_button_released

#         # Left and right axis move event
#         controller.axis_l.when_moved = on_axis_moved
#         controller.axis_r.when_moved = on_axis_moved

#         signal.pause()
# except KeyboardInterrupt:
#     pass
        