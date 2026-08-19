@echo off
REM Double-click this to run the whole Smart Traffic system.
REM
REM One USB cable leaves this PC and it goes to the ESP32 on COM6. The ESP32 is
REM the hub: it drives the LED strip and its own LCD, and reaches both Unos over
REM its own UARTs -- the lane changer on GPIO25/26, the signal heads on GPIO33
REM (to Uno pin 7). Nothing else needs plugging into the PC.
REM
REM --board tells the node which sketch is on the other end, and it matters: the
REM ESP32 is sent JSON, the Uno is sent the seven-field signal line. Name the
REM wrong one and the board sits there showing its start-up message, because
REM nothing it understands is ever sent to it. The speed follows the board
REM automatically (115200 for an esp32, 9600 for a uno).
REM
REM   --board esp32   the hub: LED strip / LCD / lane changer / signal relay  <-- set below
REM   --board uno     the traffic-light controller, plugged straight into the PC
REM
REM Driving the signal Uno over its own USB cable instead of through the hub?
REM Give the node both ports:
REM   --board esp32 --arduino-port COM6 --second-port COM4   (ESP32 on 6, Uno on 4)
REM
REM The port is COM6 here. If Device Manager shows a different number, change
REM it below -- that is the only place to change it. Close the Arduino IDE's
REM Serial Monitor first: Windows lets only one program hold a COM port, and
REM whoever gets there first wins.
REM
REM Everything after "start.py" is passed to the CV node, so you can edit this
REM line to change how it starts, e.g.  --source road.mp4  or  --no-arduino
cd /d "%~dp0"
python "start.py" --arduino-port COM6 --board esp32 %*
echo.
echo The system has stopped. Press any key to close this window.
pause > nul
