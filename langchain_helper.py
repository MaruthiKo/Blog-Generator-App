from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


def generate_blog(description, tone):
    
    repo_id = "adept/fuyu-8b" 
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5}
    )

    prompt_template_name = PromptTemplate(
        input_variables=["description","topic"],
        template = "Create a detailed blog post about {description}. The blog post should be in a {tone} tone"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    response = name_chain({"description": description, "tone":tone})
    return response

if __name__ == "__main__":
    print(generate_blog("What is Machine Learning?", "professional"))
    # print("HEllo")