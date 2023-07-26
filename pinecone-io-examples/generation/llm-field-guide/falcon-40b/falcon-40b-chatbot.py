# %% [markdown]
# # Falcon 40B Chatbot in Hugging Face and LangChain
# 
# In this notebook we'll explore how we can use the open source **Falcon-40B-Instruct** model in both Hugging Face transformers and LangChain.
# At the time of writing, this was the top ranked open source LLM according to the [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
# 
# ---
# 
# 🚨 _Note that running this on CPU is practically impossible. It will take a very long time. If running on Google Colab you go to **Runtime > Change runtime type > Hardware accelerator > GPU > GPU type > A100**. Using this notebook requires ~28GB of GPU RAM._
# 
# ---
# 
# We start by doing a `pip install` of all required libraries.

# %%
# !pip install -qU transformers accelerate einops langchain xformers bitsandbytes

# %% [markdown]
# ## Initializing the Hugging Face Pipeline

# %% [markdown]
# The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# 
# * A LLM, in this case it will be `tiiuae/falcon-40b-instruct`.
# 
# * The respective tokenizer for the model.
# 
# * A stopping criteria object.
# 
# We'll explain these as we get to them, let's begin with our model.
# 
# We initialize the model and move it to our CUDA-enabled GPU. Using Colab this can take 5-10 minutes to download and initialize the model.

# %%
from torch import cuda, bfloat16
import transformers

model_name = 'tiiuae/falcon-40b-instruct'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

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

# %% [markdown]
# The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Falcon-40B model was trained using the `falcon-40b` tokenizer, which we initialize like so:

# %%
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# %% [markdown]
# Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# %%
from transformers import StoppingCriteria, StoppingCriteriaList

# we create a list of stopping criteria
stop_token_ids = [
    tokenizer.convert_tokens_to_ids(x) for x in [
        ['Human', ':'], ['AI', ':']
    ]
]

stop_token_ids

# %% [markdown]
# We need to convert these into `LongTensor` objects:

# %%
import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

# %% [markdown]
# We can do a quick spot check that no `<unk>` token IDs (`0`) appear in the `stop_token_ids` — there are none so we can move on to building the stopping criteria object that will check whether the stopping criteria has been satisfied — meaning whether any of these token ID combinations have been generated.

# %%
from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# %% [markdown]
# Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.

# %%
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# %% [markdown]
# Confirm this is working:

# %%
res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])

# %% [markdown]
# Now to implement this in LangChain

# %%
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}"
)

llm = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# %%
print(llm_chain.predict(
    instruction="Explain to me the difference between nuclear fission and fusion."
).lstrip())

# %% [markdown]
# We still get the same output as we're not really doing anything differently here, but we have now added MTP-30B-chat to the LangChain library. Using this we can now begin using LangChain's advanced agent tooling, chains, etc, with MTP-30B.

# %% [markdown]
# ## Falcon-40B Chatbot
# 
# Using the above and LangChain we can create a conversational agent very easily. We start by initializing the conversational memory required:

# %%
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="history",  # important to align with agent prompt (below)
    k=5,
    #return_messages=True  # for conversation agent
    return_only_outputs=True  # for conversation chain
)

# %% [markdown]
# Now we initialize the conversational chain itself:

# %%
from langchain.chains import ConversationChain

chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# %%
res = chat.predict(input='hi how are you?')
res

# %% [markdown]
# Looks good so far, but there's a clear issue here, our output includes the cut off we set of `"Human:"`. Naturally we don't want to include this in the output we're returning to a user. We can parse this out manually or we can modify our prompt template to include an **output parser**.
# 
# To do this we will first need to create our output parser, which we do like so:

# %%
from langchain.schema import BaseOutputParser

class OutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        """Cleans output text"""
        text = text.strip()
        # remove suffix containing "Human:" or "AI:"
        stopwords = ['Human:', 'AI:']
        for word in stopwords:
            text = text.removesuffix(word)
        return text.strip()

    @property
    def _type(self) -> str:
        """Return output parser type for serialization"""
        return "output_parser"

parser = OutputParser()

# %% [markdown]
# Now we initialize a new prompt template, for that we need to initialize the object with a conversational prompt template, we can re-use our existing one from the conversational chain.

# %%
print(chat.prompt.template)

# %%
prompt_template = \
"""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_template,
    output_parser=parser
)

memory = ConversationBufferWindowMemory(
    memory_key="history",  # important to align with agent prompt (below)
    k=5,
    #return_messages=True  # for conversation agent
    return_only_outputs=True  # for conversation chain
)

chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
    prompt=prompt
)

# %% [markdown]
# With everything initialized we can try `predict_and_parse` which will predict for our model and then parse that prediction through the output parser we have defined.

# %%
res = chat.predict_and_parse(input='hi how are you?')
res

# %% [markdown]
# Now things are working and we don't have the messy `"Human:"` string left at the end of our returned output. Naturally we can add more logic as needed to the output parser.
# 
# We can continue the conversation to see how well Falcon 40B performs...

# %%
query = \
"""can you write me a simple Python script that calculates the circumference
of a circle given a radius `r`"""

res = chat.predict_and_parse(input=query)
res

# %%
print(res)

# %% [markdown]
# Let's try this code:

# %%
def circumference_of_circle(r):
    return 2*pi*r

circumference_of_circle(3)

# %% [markdown]
# Let's return this error back to the chatbot.

# %%
query = \
"""Using this code I get the error:

`NameError: name 'pi' is not defined`

How can I fix this?"""
res = chat.predict_and_parse(input=query)
print(res)

# %% [markdown]
# Let's try:

# %%
import math

def circumference_of_circle(r):
    return 2*math.pi*r

circumference_of_circle(3)

# %% [markdown]
# Perfect, we got the answer — not immediately but we did get it in the end. Now let's try refactoring some code to see how the model does.

# %%
def sum_numbers(n):
    total = 0
    for i in range(1, n+1):
        if i % 2 == 0:
            total += i
        else:
            total += i
    return total

# Test the function
result = sum_numbers(10)
print(result)

# %%
def sum_numbers(n):
    total = 0
    for i in range(2, n+1, 2):
        total += i
    return total

# Test the function
result = sum_numbers(10)
print(result)


# %%
query = \
"""Thanks that works! I have some code that I'd like to refactor, can you help?

The code is:

```python
def sum_numbers(n):
    total = 0
    for i in range(1, n+1):
        if i % 2 == 0:
            total += i
        else:
            total += i
    return total

# Test the function
result = sum_numbers(10)
print(result)
```
"""

res = chat.predict_and_parse(input=query)
print(res)

# %%
# --- original function ---
def sum_numbers(n):
    total = 0
    for i in range(1, n+1):
        if i % 2 == 0:
            total += i
        else:
            total += i
    return total

# Test the function
result = sum_numbers(10)
print(result)


# --- refactored function ---
def sum_numbers(n):
    return sum([i for i in range(1, n+1)])

# Test the function
result = sum_numbers(10)
print(result)

# %% [markdown]
# With that we have our Falcon-40B powered chatbot running on a single GPU using ~27.3GB of GPU RAM.
# 
# ---

# %%



