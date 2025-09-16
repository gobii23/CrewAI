from crewai import Crew, Process
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import Own_Agent
from task import Own_Task

#----------------FASTAPI---------------------
app = FastAPI()


#-----------------AGENT-----------------------
agent = Own_Agent()
manager_agent = agent.manager_agent()
automl_agent = agent.automl_agent()


#---------------TASK---------------------------
task = Own_Task()

#---------------INPUT MODEL----------------------
class automl_request(BaseModel):
    file_path : str


@app.post("/run/automl")
def run_automl(req : automl_request):
    try:
        file_path = f"data/csv_data/{req.file_path}.csv"
        automl_task = task.automl_task(automl_agent, file_path)
        automl_crew = Crew(
            agents=[automl_agent],
            tasks=[automl_task],
            manager_agent=manager_agent,
            memory=True,
            process=Process.hierarchical)
        result = automl_crew.kickoff(inputs={'file_path' : file_path})
        return {"task": "automl_task", "inputs": req.model_dump(), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


