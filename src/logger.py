import logging 
import os 
from datetime import datetime 


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE) ## this creates the log path 
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH= os.path.join(log_path,LOG_FILE) ## this creates the log file path

logging.basicConfig(
    filename=LOG_FILE_PATH, ## where to save the log files 
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__=="__main__":
    logging.info("logging has started ")