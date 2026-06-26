# # # Parallel chain WORKFLOW
# Step 1: User provides a document
# Step 2: We provide Notes and Quiz to the user
# OUTPUT: Notes + Q&A from the document provided by the user.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
groq_model = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}.",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answers from the following text \n {text}.",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document. \n Notes: {notes} \n Quiz: {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | gemini_model | parser,
        "quiz": prompt2 | groq_model | parser
    }
)

merge_chain = prompt3 | gemini_model | parser

parallel_workflow = parallel_chain | merge_chain


text = """
Cricket is a bat-and-ball game played between two teams of eleven players on a field at the center of which is a 22-yard pitch with a wicket at each end, each comprising two bails balanced on three stumps. The batting side scores runs by striking the ball bowled at the wicket with the bat, while the bowling and fielding side tries to prevent this and dismiss each player (so they are "out"). Means of dismissal include being bowled, when the ball hits the stumps and dislodges the bails, and by the fielding side catching the ball after it is hit by the bat, but before it hits the ground. When ten players have been dismissed, the innings ends and the teams swap roles. The game is adjudicated by two umpires, aided by a third umpire and match referee in international matches. They communicate with two off-field scorers who record the match's statistical information.
"""

result =parallel_workflow.invoke({'text': text})

print(result)

