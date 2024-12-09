# icdt (I can't do it)

## Prerequisites
KUKA LBR Med 7 R800 connected through Ethernet with ip 172.31.1.147

Host ip 172.31.1.148, with netmask 24

Franka FR3 connected through Ethernet with ip 172.16.0.2

Host ip 172.16.0.1, with netmask 24

Put Med 7 in AUT mode and run [LBRServer](https://github.com/lbr-stack/fri/tree/fri-2.5/server_app) when at step "Launch robot infrastructure"

Activate FCI in Franka FR3

Find your recording device's name by using arecord -l

Modify the name on line 27 in whisper_node/whisper_ros/task_transcription.py

## Usage

### OPENAI_KEY
Change all occurances of 'OPENAI_KEY' in icdt_wrapper/icdt_wrapper folder
```
OPENAI_KEY = <YOUR OPENAI_KEY IN STRING>
```

### Build the docker image
```
cd icdt
./docker/tekkneeca.sh build
```

### Run the docker image
```
./docker/tekkneeca.sh run
```

### (Optional) If ros_ws did not build successfully
```
tek_install
```

### Launch robot infrastructure
```
ros2 launch icdt_wrapper icdt_duet_real.launch.py
```

### Launch perception service
```
ros2 launch perception_server detection_services.launch.py
```

### (Option 1) Run LLM interface
```
ros2 run icdt_wrapper duet_interface.py
```

### (Option 2) Run Whisper LLM interface
```
ros2 run whisper_ros task_transcription 
ros2 run icdt_wrapper whisper_interface.py
```