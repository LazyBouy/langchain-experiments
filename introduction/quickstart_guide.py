import inspect
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain

# Load environment variables
load_dotenv(find_dotenv())

# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------

llm = OpenAI(model_name="text-davinci-003", temperature=0)
prompt = "Write a poem about python and ai"
print(llm(prompt))


# --------------------------------------------------------------
# Prompt Templates: Manage prompts for LLMs
# --------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

prompt.format(product="Smart Apps using Large Language Models (LLMs)")

# --------------------------------------------------------------
# Chains: Combine LLMs and prompts in multi-step workflows
# --------------------------------------------------------------

llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI Chatbots for Dental Offices"))


# --------------------------------------------------------------
# Agents: Dynamically Call Chains Based on User Input
# --------------------------------------------------------------


llm = OpenAI()

get_all_tool_names()
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Now let's test it out!
result = agent.run(
    "In what year was python released and who is the original creator? Multiply the year by 3"
)
print(result)


# --------------------------------------------------------------
# Memory: Add State to Chains and Agents
# --------------------------------------------------------------

llm = OpenAI()
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(
    input="I'm doing well! Just having a conversation with an AI."
)
print(output)


# --------------------------------------------------------------
# Math: LLMMathchain
# --------------------------------------------------------------

llm_math = LLMMathChain(llm=llm, verbose=True)
print(llm_math.prompt.template)

print(inspect.getsource(llm_math._call))


# --------------------------------------------------------------
# Initialize Falcon-40B-instruct
# --------------------------------------------------------------
from torch import cuda, bfloat16
import transformers

model_name = 'tiiuae/falcon-40b-instruct'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto'
)
model.eval()
print(f"Model loaded on {device}")
