# --------------------------------------------------------------
# Hugging Face Hub
# --------------------------------------------------------------
from dotenv import find_dotenv, load_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Load environment variables
load_dotenv(find_dotenv())

question = "Who won the FIFA World Cup in the year 1994?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, 
                        input_variables=["question"]
                        )

repo_id = "tiiuae/falcon-7b-instruct"

llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))