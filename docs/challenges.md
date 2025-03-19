### **Phase 1: Day 1 to 15 - Introduction, Domain Familiarization, and Environment Setup**

#### **Topics Covered**
1. **Automated Trading & Web3 Integration**
   - Introduction to the basics of automated trading, covering:
     - Challenges like market volatility, latency, and data inconsistencies.
     - Importance of data-driven decision-making for trading strategies.
   - Overview of data sources:
     - **On-chain data**: Blockchain explorers (e.g., Etherscan, BSCScan) and DeFi protocols.
     - **Off-chain data**: Centralized exchange (CEX) APIs (e.g., Binance, OKX) and news feeds.
   - **Key Challenges:**
     - Data noise, API rate limits, and latency in data ingestion.

2. **LLM Agent Fundamentals**
   - Role of Large Language Models (LLMs) in financial trading:
     - Prompt engineering to generate trade signals from unstructured data.
     - Addressing common pitfalls like noisy data and inference latency.

3. **Environment Setup**
   - Setting up development environments:
     - Installing Python, Node.js, Docker, and Git.
     - Installing CUDA drivers (if targeting GPU acceleration).
   - Introduction to Docker and CI/CD pipelines.

---

#### **Hands-On Tasks**
1. **Project Scaffold**
   - Created a GitHub repository with an initial project scaffold, including:
     - A "Hello Trading" simulation program to verify the development setup.
   - Organized the folder structure for maintainability and scalability.

2. **Docker Setup**
   - Installed and configured Docker with a **CUDA-enabled base image** to support GPU workloads.
   - Developed a `Dockerfile` to containerize the project.

3. **CI/CD Pipeline**
   - Set up a continuous integration/continuous deployment pipeline using **GitHub Actions**:
     - Automated tests for code validation.
     - Integrated Docker builds within the CI/CD workflow.

4. **Documentation**
   - Drafted documentation to summarize:
     - Trading domain challenges like volatility, noisy data, and latency.
     - Role of AI/LLM agents in addressing these challenges.

---

#### **Deliverables**
1. **GitHub Repository**
   - A public GitHub repository with:
     - A well-documented project scaffold.
     - An operational `Dockerfile`.
     - Configured CI/CD pipeline using GitHub Actions.

2. **Documentation**
   - Summarized the challenges in automated trading.
   - Explained the importance and best practices of LLM integration.

3. **Introductory Blog Post**
   - Outlined the project vision and technical roadmap.

---

#### **Conclusion**
Phase 1 (Day 1-15) laid the foundation for the project:
- Curated learning materials on automated trading, LLMs, and Web3.
- Set up the project scaffold and environment for seamless development.
- Delivered initial documentation, Docker setup, and CI/CD integration to ensure a robust workflow.
