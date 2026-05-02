# ⚖️ Legal AI Assistant

An advanced, multi-agent AI application designed to analyze Indian statutory laws, regulations, and landmark case precedents. Built with Streamlit, LangChain, FAISS, and Groq.

## 🚀 Hugging Face Deployment
* **Live App:** [https://huggingface.co/spaces/rajann/legalapp](https://huggingface.co/spaces/rajann/legalapp)

## 📚 Supported Documents
The assistant includes vector representations for the following core corpora:
* **Statutory Acts:** Companies Act (2013), Consumer Protection Act (2019), Indian Contract Act (1872), Limited Liability Partnership (LLP) Act (2008), Sale of Goods Act (1930).
* **Landmark Precedents:** * *Vodafone International Holdings B.V. vs Union of India*
  * *Commissioner of Wealth-Tax vs Abdul Hussain Mulla Muhammad Ali*
  * *M/s K Home Appliances vs M/s Marvs Travel India Pvt. Ltd. & Ors.*
  * *M/s Hindustan Coca-Cola Beverages Pvt. Ltd. vs Cce Visakhapatnam*
  * *Ramesh Yadav vs The New India Assurance Company Limited*
* **Regulatory Materials:** Framework conditions and guidelines for the Competition Commission of India (CCI) and the Warehousing Development and Regulatory Authority (WDRA).

## 🚀 How to Use

1. **Access:** Open the application in your browser and use the **App Settings & Documentation Info** panel to review the active corpus.
2. **Preset Queries:** Click any of the predefined examples (e.g., *Free Consent in Contract Act*, *Vodafone International Holdings*) to populate the query field automatically.
3. **Ask:** Type your specific legal or regulatory question into the text input box and click **Ask** to run the multi-agent reasoning pipeline.

## ⚙️ Project Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/legalapp.git](https://github.com/your-username/legalapp.git)
   cd legalapp
