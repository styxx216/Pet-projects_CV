
### Project Description:

Video Traffic Analytics: Real-time Detection, Tracking, and Counting of Moving Objects Crossing a Control Line

### Project Overview
This project implements a real-time object counting system using the YOLO (You Only Look Once) object detection model. The system detects, tracks, and counts moving objects that cross a designated control line. It supports various use cases such as:

Counting people on an escalator
Counting cars on a road
Counting vaccine vials on a conveyor belt (trained on a custom dataset)

### Features
Real-time object detection and counting: Utilizes the YOLO model to count objects crossing a control line in live video feeds.
Pre-trained YOLO model: Supports generic objects like people and vehicles without additional training.
Custom dataset support: Allows custom object counting, demonstrated with vaccine vials on a conveyor belt.
Efficient tracking: Ensures accurate counting even in fast-moving environments.

### Examples
#### People on escalator: 

![ezgif-6-716de5b444](https://github.com/styxx216/CV/assets/38997882/d941eef0-31e3-46b3-ab6b-1dc849d1f13b)
* **Goal**: Count the number of people using the escalator in real-time.
* **Model**: Pre-trained YOLO model.
* **Description**: The system detects and tracks people as they pass a control line on an escalator, providing a live count.

#### Cars on the road: 

![ezgif-7-7c9318f4d0](https://github.com/styxx216/CV/assets/38997882/41cadb65-813b-46f8-9ffa-ebdfac15eee8)

* **Goal**: Real-time detection and counting of cars passing a specific point on the road.
* **Model**: Pre-trained YOLO model.
* **Description**: The system identifies and counts vehicles as they cross a control line on a busy road.

#### Vials on the conveyor belt:

![ezgif-7-52b116c426](https://github.com/styxx216/CV/assets/38997882/9f75701f-1024-44b9-af57-815668e26254)

* **Goal**: Count vaccine vials in a production line for inventory management.
* **Model**: YOLO model trained on a custom dataset.
* **Description**: Custom-trained YOLO model detects and counts vaccine vials in real time as they move along a conveyor belt.
