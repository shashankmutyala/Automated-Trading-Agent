# Setup Instructions
## **Days(1-15)**
## Prerequisites
Before setting up the project, ensure the following are installed on your system:
- **Python 3.9+**
- **Node.js and npm**
- **Docker**
- **NVIDIA Drivers (if using GPU acceleration)**

---

## Step-by-Step Setup
### 1. **Clone the Repository**
```bash
git clone https://github.com/shashankmutyala/Automated-Trading-Agent.git
cd Automated-Trading-Agent
```
### 2. **Set Up the Backend**
Install Python Dependencies:
```bash
pip install -r requirements.txt
```
## 3. **Set Up Docker**
Build the Docker Image
```bash
docker build -t automated-trading-agent .
```
Run the Docker Container
```bash
docker run --rm --gpus all automated-trading-agent
```
## 4. **CI/CD Pipeline Setup**

The Automated Trading Agent uses GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD). The workflow is defined in .github/workflows/ci-cd.yml and ensures:

✅ Automated Testing - Runs unittest to verify core functionalities before deployment.
✅ Docker Build & Deployment - Ensures a stable containerized deployment.
✅ Code Quality Checks - Linting and formatting checks for clean code.

🚀 Running the CI/CD Pipeline

The CI/CD pipeline is triggered automatically on:

Push to main branch

Pull requests to main

To manually trigger the workflow, navigate to the GitHub Actions tab in your repository and run the latest workflow.

