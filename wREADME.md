# Intelligent Pesticide Sprinkling System (SIH 2025)
### Problem Statement ID: 25015  
### Team: Twin Spark  
### Institution: Integral University, Lucknow  

---

##  1. What Our Project Does  
This project implements an **AI-powered pesticide sprinkling system** that detects plant infection levels using a camera mounted on a **rover and drone**.  
Based on the infection percentage, the system **automatically decides how much pesticide to spray**, reducing cost, chemical usage, and farmer effort.

It includes:
- A **Machine Learning model** with 39 disease classes (PlantVillage dataset).
- A **FastAPI backend** for image prediction & infection analysis.
- A **control system for a rover and drone** equipped with cameras.
- A **3-role Dashboard UI**: Farmers, Government, Agro-chemical Researchers.

---

##  2. Problem Statement & Solution  

### **Problem Summary**
Farmers often spray pesticides manually, which leads to:
- Excessive use of chemicals (higher cost + health hazards)
- Uneven spraying
- Delayed disease detection
- Environmental damage

### **Our Solution**
We built an **intelligent, automated** pesticide spraying system that:
- Detects disease early using AI and computer vision  
- Calculates **infection percentage** and recommends action  
- Sprays pesticide **only where required**, making it **cost-effective**  
- Provides dashboards for farmers, Govt, and Agro-industry researchers  
- Uses Drone + Rover cameras for large-area and ground-level scanning  

This makes the entire system:

More effective for farmers  
More affordable  
Environment-friendly  
Scientifically supported  

---

##  3. Features  

###  **AI Model**
- Trained on **61,000+ leaf images**  
- 39 disease classes  
- Outputs **infection %** and recommended action  

###  **Robotics**
- Rover with camera for close-up scanning  
- Drone with aerial camera for field-wide scanning  
- Automated pesticide spray control  

###  **Intelligent Decision System**
- Computes infection percentage  
- Determines Low / Medium / High spray  
- Reduces unnecessary pesticide use  

###  **3 Dashboards**
1. **Farmer Dashboard** – Simple UI for disease detection & spray control  
2. **Government Dashboard** – Crop disease and Pesticides usage monitoring & analytics of various farms and Responding to red alerts  
3. **Agro-chemical Industry Dashboard** – monitoring Pesticide Usage and Detecting highly effective pesticides  

###  **Backend API**
- FastAPI-based  
- Endpoints for image prediction  
- Integrates ML model & robotics  

---

##  4. Technologies Used  

| Component | Technology |
|----------|------------|
| ML Model | PyTorch, TorchVision, Python |
| Backend | FastAPI, Uvicorn,  Firebase |
| Frontend | React / HTML / CSS Tailwind / JS |
| Robotics | ESP32, Sensors, Actuators, ESP32 cam,  Motor Drivers, Drone Camera |
| Dataset | PlantVillage |
| Deployment | Local server / edge device |

---

##  5. Steps to Install & Run  

### **1️⃣ Install Dependencies**

pip install -r requirements.txt


### **2️⃣ Run Backend**
python backend/backend.py

Then open:
http://127.0.0.1:8000, http://127.0.0.1:8000/docs


### **3️⃣ Run Frontend**

cd frontend

npm install

npm start


### **4️⃣ Upload Image & Test Prediction**
Use `/predict` or the dashboard upload interface.

---

##  6. Required Environment Variables (If Needed)
MODEL_PATH=checkpoints/epoch_30.pth
CLASSES_FILE=data/classes.txt
IMG_SIZE=224


---

##  7. Screenshots / Sample Inputs
(`Sample-image`/`assets/screenshots/`)

---

##  8. Team Members  
All members are from **Integral University**.

| Name | Role |
|------|------|
| **Intakhab Nabi** | ML model, training, inference, drone-development |
| **Akhlaque Nabi** | Backend, API |
| **Abdul Saboor** | Database, research, drone-development |
| **Anamta Rizvi** | Frontend (Farmer & Govt UI) |
| **Aasiya Faseeh** | Frontend (Agro-industry UI) |
| **Abdul Ali** | Robotics (Rover + Drone Integration) |

---

##  9. GitHub Repository Link  
<https://github.com/Raaaaahil/Twin-Spark_SIH2025>

---

##  10. Why This Solution is Cost-Effective & Farmer-Friendly  
- AI ensures **minimum pesticide usage**, reducing farmer cost  
- Bluetooth/IoT-controlled spraying reduces labor cost  
- Early detection prevents large-scale crop damage  
- Dashboards help govt & companies plan better  
- Rover + Drone scanning eliminates manual checking  
- AI-driven targeted spraying: On-deviceML with GPS-tagged detection enableshighly selective treatment, reducingpesticide usage by up to 70%.
- Ultra-low prototype cost: Bill ofmaterials under Rs.1500 makes the systemaffordable and enables fast adoption.
- Designed for Indian fields: Hardwareand enclosures account for heat, dust,humidity and uneven terrain to improve real-world reliability.
- Lifecycle monitoring: Effective fromseedling through harvest with historical logsto inform treatment timing.
- Modular and scalable: Supports roverdeployments for small plots and droneintegration for larger areas.
- Farmer-centric UX: Simple dashboard,auto/manual modes and minimal setupreduce training overhead.

##  9. Pipeline flow Chart Link  
<https://drive.google.com/drive/folders/1qRDIgytoAii22Oew2gzjxMW7UnGS1hJb?usp=sharing>