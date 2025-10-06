import os
# import sqlite3
import pandas as pd
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from flask import Flask, jsonify, request, redirect, Response
from dotenv import load_dotenv
# import subprocess
# import threading
import time
from datetime import datetime
import sys
# import webbrowser
import shutil
from flask_cors import CORS
import json
import uuid
# from collections import OrderedDict

# Load environment variables
load_dotenv("./dev/webui.env")

# --- Configuration with proper defaults ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
WEBUI_PORT = os.getenv("WEBUI_PORT", "3000")
API_PORT = "1898" # int(os.getenv("API_PORT", "8787"))
WEBUI_NAME = os.getenv("WEBUI_NAME", "Vanna SQL Assistant")
WEBUI_DESCRIPTION = os.getenv("WEBUI_DESCRIPTION", "SQL Query Assistant powered by Vanna")
LLM_TEMPERATURE=float(os.getenv("TEMPERATURE",0.7))

print("=== Configuration ===")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'NOT SET'}...")
print(f"OPENAI_MODEL: {OPENAI_MODEL}")
print(f"WEBUI_PORT: {WEBUI_PORT}")
print(f"API_PORT: {API_PORT}")
print(f"WEBUI_NAME: {WEBUI_NAME}")
print(f"WEBUI_DESCRIPTION: {WEBUI_DESCRIPTION}")
print("====================")

# # --- Create database ---
# def create_database():
#     db_path = "dummy_sales.db"
#     if not os.path.exists(db_path):
#         conn = sqlite3.connect(db_path)
#         df = pd.DataFrame([
#             ("Alice", "Laptop", 1, 1200.00, "2024-01-10"),
#             ("Bob", "Phone", 2, 600.00, "2024-01-15"),
#             ("Charlie", "Tablet", 1, 300.00, "2024-02-05"),
#             ("Alice", "Monitor", 2, 200.00, "2024-02-20"),
#             ("Bob", "Keyboard", 3, 50.00, "2024-03-02"),
#         ], columns=["customer", "product", "quantity", "price", "sale_date"])
#         df.to_sql("sales", conn, if_exists="replace", index=False)
#         conn.commit()
#         conn.close()
#     return db_path
path_prefix='models/hr'
import logging
# Create a logger
# file_name = str(datetime.now()).replace(":",";")

# file_handler = logging.FileHandler(os.path.join(".",path_prefix,"debug",file_name+".log"))  # Log file name



logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Line %(lineno)d - %(message)s')
handler1 = logging.FileHandler(os.path.join(path_prefix,'app.log'))
handler1.setFormatter(formatter)
logger.addHandler(handler1)


logger.info("Log is successfully instantiated.")






# --- Reset ChromaDB if needed ---
def reset_chromadb():
    chroma_path = os.path.join(path_prefix,"chroma.sqlite3")
    if os.path.exists(chroma_path):
        try:
            backup_path = os.path.join(path_prefix,f"chroma.sqlite3.bak.{int(time.time())}")
            shutil.move(chroma_path, backup_path)
            logger.info(f"Backed up existing ChromaDB to {backup_path}")
        except Exception as e:
            logger.error(f"Error backing up ChromaDB: {e}")
            try:
                os.remove(chroma_path)
                logger.info(f"Deleted existing ChromaDB file")
            except Exception as e:
                logger.error(f"Error deleting ChromaDB file: {e}")
                pass
anchor = "EVAL"
# --- Enhanced Vanna class with proper data visibility ---
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        if config is None:
            config = {}

        config["chroma_client_settings"] = {
            "is_persistent": False,
            "persist_directory": "."
        }

        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

        # Override the allow_llm_to_see_data setting
        self.allow_llm_to_see_data = True

    def generate_sql(self, question: str, allow_llm_to_see_data: bool = True, **kwargs) -> str:
        """Override to ensure data visibility is always enabled"""
        # Force allow_llm_to_see_data to True
        return super().generate_sql(question, allow_llm_to_see_data=True, **kwargs)

    def ask(self, question: str, allow_llm_to_see_data: bool = True, **kwargs) -> str:
        """Override ask method to ensure data visibility"""
        return super().ask(question, allow_llm_to_see_data=True, **kwargs)
    # def system_message(self,message):
    #     custom_message = ""
    #     logger.debug("System message IMPORTANT")
    #     message = message  + custom_message
    #     logger.debug(message)
    #     logger.debug("END OF OVERRIDEN MESSAGE.")
    #     return super().system_message(message)

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        results = super().get_similar_question_sql(question, **kwargs)
        logger.debug("get_similar_question_sql")
        for similar_question_sql in results:
            logger.debug(json.dumps(similar_question_sql,indent=3))
        return results

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        results = super().get_related_ddl(question, **kwargs)
        logger.debug("get_related_ddl")
        for related_ddl in results:
            logger.debug(json.dumps(related_ddl,indent=3))
        return results  

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        results = super().get_related_documentation( question, **kwargs)
        logger.debug("get_related_documentation")
        for related_documentation in results:
            logger.debug(json.dumps(related_documentation,indent=3))
        return results  

    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Example:
        ```python
        vn.submit_prompt(
            [
                vn.system_message("The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."),
                vn.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        results = super().submit_prompt( prompt, **kwargs)
        logger.debug("submit_prompt")
        logger.debug(json.dumps(results,indent=3))
        return results   


# --- Initialize Vanna ---
print("Initializing Vanna...")
reset_chromadb()

vn = None
if OPENAI_API_KEY:
    vn = MyVanna(config={
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "allow_llm_to_see_data": True,
        # the n_results is use to to indicate how much related embeddings will be retrieved to be sent to the AI model.
        "n_results":2,
        "temperature":LLM_TEMPERATURE,
        # this path is used to create the cromadb
        "path":path_prefix
    })

    logger.debug(f"LLM temperature is {LLM_TEMPERATURE}")

    # Set the allow_llm_to_see_data attribute directly
    vn.allow_llm_to_see_data = True

    # db_path = create_database()
    # vn.connect_to_sqlite(db_path)
    vn.connect_to_postgres(host='3.29.246.202', dbname='odoo', user='odoo18', password='odoo18', port='5432')
    logger.info("Database connected!")

    ddls = [
    """
        CREATE TABLE public.hr_employee
    (
    id integer DEFAULT nextval('hr_employee_id_seq'::regclass),
    resource_id integer NOT NULL,
    company_id integer NOT NULL,
    resource_calendar_id integer,
    message_main_attachment_id integer,
    color integer,
    department_id integer,
    job_id integer,
    address_id integer,
    work_contact_id integer,
    work_location_id integer,
    user_id integer,
    parent_id integer,
    coach_id integer,
    address_home_id integer,
    country_id integer,
    children integer,
    country_of_birth integer,
    bank_account_id integer,
    km_home_work integer,
    departure_reason_id integer,
    create_uid integer,
    write_uid integer,
    name character varying ,
    job_title character varying ,
    work_phone character varying ,
    mobile_phone character varying ,
    work_email character varying ,
    employee_type character varying  NOT NULL,
    gender character varying ,
    marital character varying ,
    spouse_complete_name character varying ,
    place_of_birth character varying ,
    ssnid character varying ,
    sinid character varying ,
    identification_id character varying ,
    passport_id character varying ,
    permit_no character varying ,
    visa_no character varying ,
    certificate character varying ,
    study_field character varying ,
    study_school character varying ,
    emergency_contact character varying ,
    emergency_phone character varying ,
    barcode character varying ,
    pin character varying ,
    spouse_birthdate date,
    birthday date,
    visa_expire date,
    work_permit_expiration_date date,
    departure_date date,
    additional_note text ,
    notes text ,
    departure_description text ,
    active boolean,
    work_permit_scheduled_activity boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT hr_employee_pkey PRIMARY KEY (id),
    CONSTRAINT hr_employee_barcode_uniq UNIQUE (barcode),
    CONSTRAINT hr_employee_user_uniq UNIQUE (user_id, company_id),
    CONSTRAINT hr_employee_address_home_id_fkey FOREIGN KEY (address_home_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_address_id_fkey FOREIGN KEY (address_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_bank_account_id_fkey FOREIGN KEY (bank_account_id)
        REFERENCES public.res_partner_bank (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_coach_id_fkey FOREIGN KEY (coach_id)
        REFERENCES public.hr_employee (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_country_id_fkey FOREIGN KEY (country_id)
        REFERENCES public.res_country (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_country_of_birth_fkey FOREIGN KEY (country_of_birth)
        REFERENCES public.res_country (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_department_id_fkey FOREIGN KEY (department_id)
        REFERENCES public.hr_department (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_departure_reason_id_fkey FOREIGN KEY (departure_reason_id)
        REFERENCES public.hr_departure_reason (id) MATCH SIMPLE,
    CONSTRAINT hr_employee_job_id_fkey FOREIGN KEY (job_id)
        REFERENCES public.hr_job (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id)
        REFERENCES public.ir_attachment (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_parent_id_fkey FOREIGN KEY (parent_id)
        REFERENCES public.hr_employee (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_resource_calendar_id_fkey FOREIGN KEY (resource_calendar_id)
        REFERENCES public.resource_calendar (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_resource_id_fkey FOREIGN KEY (resource_id)
        REFERENCES public.resource_resource (id) MATCH SIMPLE,
    CONSTRAINT hr_employee_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_work_contact_id_fkey FOREIGN KEY (work_contact_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_work_location_id_fkey FOREIGN KEY (work_location_id)
        REFERENCES public.hr_work_location (id) MATCH SIMPLE
        ,
    CONSTRAINT hr_employee_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE    
    )

    """,
    """
    CREATE TABLE public.hr_department
    (
    id integer NOT NULL DEFAULT nextval('hr_department_id_seq'::regclass),
    message_main_attachment_id integer,
    company_id integer,
    parent_id integer,
    manager_id integer,
    color integer,
    master_department_id integer,
    create_uid integer,
    write_uid integer,
    name character varying  NOT NULL,
    complete_name character varying ,
    parent_path character varying ,
    note text ,
    active boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT hr_department_pkey PRIMARY KEY (id),
    CONSTRAINT hr_department_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
    ,
    CONSTRAINT hr_department_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
    ,
    CONSTRAINT hr_department_manager_id_fkey FOREIGN KEY (manager_id)
        REFERENCES public.hr_employee (id) MATCH SIMPLE
    ,
    CONSTRAINT hr_department_master_department_id_fkey FOREIGN KEY (master_department_id)
        REFERENCES public.hr_department (id) MATCH SIMPLE
    ,
    CONSTRAINT hr_department_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id)
        REFERENCES public.ir_attachment (id) MATCH SIMPLE
    ,
    CONSTRAINT hr_department_parent_id_fkey FOREIGN KEY (parent_id)
        REFERENCES public.hr_department (id) MATCH SIMPLE
    ,
    CONSTRAINT hr_department_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE)
        """,
        """
        CREATE TABLE public.hr_work_location
    (
    id integer  DEFAULT nextval('hr_work_location_id_seq'::regclass),
    company_id integer ,
    address_id integer ,
    create_uid integer,
    write_uid integer,
    name character varying COLLATE pg_catalog."default" ,
    location_type character varying COLLATE pg_catalog."default" ,
    location_number character varying COLLATE pg_catalog."default",
    active boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT hr_work_location_pkey PRIMARY KEY (id),
    CONSTRAINT hr_work_location_address_id_fkey FOREIGN KEY (address_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE,
    CONSTRAINT hr_work_location_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE,
    CONSTRAINT hr_work_location_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE,
    CONSTRAINT hr_work_location_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
    )
        """,
        """
        CREATE TABLE public.hr_contract_type
    (
    id integer NOT NULL DEFAULT nextval('hr_contract_type_id_seq'::regclass),
    sequence integer,
    country_id integer,
    create_uid integer,
    write_uid integer,
    code character varying COLLATE pg_catalog."default",
    name jsonb NOT NULL,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT hr_contract_type_pkey PRIMARY KEY (id),
    CONSTRAINT hr_contract_type_country_id_fkey FOREIGN KEY (country_id)
        REFERENCES public.res_country (id) MATCH SIMPLE,
    CONSTRAINT hr_contract_type_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE,
    CONSTRAINT hr_contract_type_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
    )
        """,
        """
    CREATE TABLE public.hr_job
    (   
    id integer NOT NULL DEFAULT nextval('hr_job_id_seq'::regclass),
    sequence integer,
    expected_employees integer,
    no_of_employee integer,
    no_of_recruitment integer,
    department_id integer,
    company_id integer,
    contract_type_id integer,
    create_uid integer,
    write_uid integer,
    name jsonb ,
    description text COLLATE pg_catalog."default",
    requirements text COLLATE pg_catalog."default",
    active boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT hr_job_pkey PRIMARY KEY (id),
    CONSTRAINT hr_job_name_company_uniq UNIQUE (name, company_id, department_id),
    CONSTRAINT hr_job_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE,
    CONSTRAINT hr_job_contract_type_id_fkey FOREIGN KEY (contract_type_id)
        REFERENCES public.hr_contract_type (id) MATCH SIMPLE,
    CONSTRAINT hr_job_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE,
    CONSTRAINT hr_job_department_id_fkey FOREIGN KEY (department_id)
        REFERENCES public.hr_department (id) MATCH SIMPLE,
    CONSTRAINT hr_job_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE,
    CONSTRAINT hr_job_no_of_recruitment_positive CHECK (no_of_recruitment >= 0)
    )
        """
        ]
    for ddl in ddls:
        try:
            vn.train(ddl=ddl)
        except Exception as e:
            logger.error(e)
            logger.error(f"Error training the ddls specifically ddl {ddl}.")
    # Train Vanna with your database schema and sample data
    try:
        ## Hr department table
        vn.train(question="What are the hr departments, show some of stored hr departments",sql="select name from hr_department")
        vn.train(question="What is meant by the R&D USA department",sql="select complete_name from hr_department where name = 'R&D USA';")
        vn.train(question="Is the R&D USA department working?, what is the status of the R&D USA department?",sql="select active from hr_department where lower(name) = lower('R&D USA') or lower(complete_name) = lower('R&D USA');")
        
        ## hr_contract_type
        vn.train(question="What are the different contract types available?", sql="SELECT id, code, name FROM public.hr_contract_type;")
        vn.train(question="How many different contract types are defined?", sql="SELECT COUNT(*) FROM public.hr_contract_type;")
        # intead of using the id 1, we can join this to the users table
        vn.train(question="Which contract types were created by user ID 1?", sql="SELECT * FROM public.hr_contract_type WHERE create_uid = 1;")
        vn.train(question="List contract types created after September 1st, 2025.", sql="SELECT * FROM public.hr_contract_type WHERE create_date > '2025-09-01';")
        vn.train(question="Which contract types have the code 'Permanent'?", sql="SELECT * FROM public.hr_contract_type WHERE code = 'Permanent';")
        vn.train(question="Which users have modified contract types?", sql="SELECT DISTINCT write_uid FROM public.hr_contract_type WHERE write_uid IS NOT NULL;")
        vn.train(question="Are there any contract types with missing country information?", sql="SELECT * FROM public.hr_contract_type WHERE country_id IS NULL;")
        vn.train(question="What is the JSON name for contract type 'Full-Time'?", sql="SELECT name FROM public.hr_contract_type WHERE code = 'Full-Time';")
        vn.train(question="Which contract types were last modified on September 2, 2025?", sql="SELECT * FROM public.hr_contract_type WHERE write_date::date = '2025-09-02';")
        vn.train(question="What are the earliest and latest contract creation dates?", sql="SELECT MIN(create_date) AS earliest, MAX(create_date) AS latest FROM public.hr_contract_type;")
        vn.train(question="How many contract types have the same create and write date?", sql="SELECT COUNT(*) FROM public.hr_contract_type WHERE create_date = write_date;")
        vn.train(question="List contract types where the name JSON includes 'Seasonal'.", sql="SELECT * FROM public.hr_contract_type WHERE name::text ILIKE '%Seasonal%';")
        
        ## hr_employee
        vn.train(question="How many employees are currently active?", sql="SELECT COUNT(*) FROM public.hr_employee WHERE active = true;")
        vn.train(question="List all employees along with their job titles.", sql="SELECT name, job_title FROM public.hr_employee;")
        vn.train(question="Which employees have the job title 'Marketing and Community Manager'?", sql="SELECT * FROM public.hr_employee WHERE job_title = 'Marketing and Community Manager';")
        vn.train(question="What are the names and emails of all employees?", sql="SELECT name, work_email FROM public.hr_employee;")
        vn.train(question="Which employees do not have a mobile phone listed?", sql="SELECT * FROM public.hr_employee WHERE mobile_phone IS NULL OR mobile_phone = '';")
        vn.train(question="What is the total number of employees per department?", sql="SELECT department_id, COUNT(*) FROM public.hr_employee GROUP BY department_id;")
        vn.train(question="List all employees with their birthday and marital status.", sql="SELECT name, birthday, marital FROM public.hr_employee;")
        vn.train(question="Find employees whose visa has expired.", sql="SELECT * FROM public.hr_employee WHERE visa_expire IS NOT NULL AND visa_expire < CURRENT_DATE;")
        vn.train(question="Get the list of employees along with their countries of birth.", sql="SELECT name, country_of_birth FROM public.hr_employee;")
        vn.train(question="Which employees are not flexible?", sql="SELECT * FROM public.hr_employee WHERE is_flexible = false;")
        vn.train(question="List employees who have left the company.", sql="SELECT * FROM public.hr_employee WHERE departure_date IS NOT NULL;")
        vn.train(question="What are the different types of employee types?", sql="SELECT DISTINCT employee_type FROM public.hr_employee;")
        vn.train(question="How many employees have missing job titles?", sql="SELECT COUNT(*) FROM public.hr_employee WHERE job_title IS NULL OR job_title = '';")
        vn.train(question="Which employees have a work email ending with '@example.com'?", sql="SELECT * FROM public.hr_employee WHERE work_email ILIKE '%@example.com';")
        vn.train(question="Show the full list of employees and their coaches.", sql="SELECT e.name AS employee_name, c.name AS coach_name FROM public.hr_employee e LEFT JOIN public.hr_employee c ON e.coach_id = c.id;")
        vn.train(question="Find employees with missing barcode information.", sql="SELECT * FROM public.hr_employee WHERE barcode IS NULL OR barcode = '';")
        vn.train(question="How many employees were added on January 1, 2010?", sql="SELECT COUNT(*) FROM public.hr_employee WHERE create_date::date = '2010-01-01';")
        vn.train(question="Find employees with an emergency contact specified.", sql="SELECT * FROM public.hr_employee WHERE emergency_contact IS NOT NULL AND emergency_contact != '';")
        vn.train(question="Which employees have been updated most recently?", sql="SELECT * FROM public.hr_employee ORDER BY write_date DESC")
        vn.train(question="Get employees whose work location is ID 4.", sql="SELECT * FROM public.hr_employee WHERE work_location_id = 4;")

        # hr_work_location
        vn.train(question="What are all the available work locations?", sql="SELECT id, name FROM public.hr_work_location;")
        vn.train(question="How many work locations are currently active?", sql="SELECT COUNT(*) FROM public.hr_work_location WHERE active = true;")
        vn.train(question="Which work locations are of type 'office'?", sql="SELECT * FROM public.hr_work_location WHERE location_type = 'office';")
        vn.train(question="What are the names and types of all work locations?", sql="SELECT name, location_type FROM public.hr_work_location;")
        vn.train(question="Which user created each work location?", sql="SELECT id, name, create_uid FROM public.hr_work_location;")
        vn.train(question="What are the different types of work locations defined?", sql="SELECT DISTINCT location_type FROM public.hr_work_location;")
        vn.train(question="Which work location was created most recently?", sql="SELECT * FROM public.hr_work_location ORDER BY create_date DESC LIMIT 1;")
        vn.train(question="List work locations with the word 'Building' in their name.", sql="SELECT * FROM public.hr_work_location WHERE name ILIKE '%Building%';")
        vn.train(question="What is the total number of work locations per company?", sql="SELECT company_id, COUNT(*) FROM public.hr_work_location GROUP BY company_id;")
        vn.train(question="Which locations do not have a location number specified?", sql="SELECT * FROM public.hr_work_location WHERE location_number IS NULL OR location_number = '';")
        vn.train(question="List all work locations ordered by their name.", sql="SELECT * FROM public.hr_work_location ORDER BY name;")
        vn.train(question="Which work locations were last updated on September 2, 2025?", sql="SELECT * FROM public.hr_work_location WHERE write_date::date = '2025-09-02';")
        vn.train(question="Which work locations were created by user ID 1?", sql="SELECT * FROM public.hr_work_location WHERE create_uid = 1;")
        vn.train(question="Show all work locations and their associated addresses.", sql="SELECT id, name, address_id FROM public.hr_work_location;")
        vn.train(question="Are there any duplicate work location names?", sql="SELECT name, COUNT(*) FROM public.hr_work_location GROUP BY name HAVING COUNT(*) > 1;")
        vn.train(question="What is the location type of 'Building 1, Second Floor'?", sql="SELECT location_type FROM public.hr_work_location WHERE name = 'Building 1, Second Floor';")
        vn.train(question="Which work locations have null values for address_id?", sql="SELECT * FROM public.hr_work_location WHERE address_id IS NULL;")
        vn.train(question="List the earliest and latest creation dates for work locations.", sql="SELECT MIN(create_date) AS earliest, MAX(create_date) AS latest FROM public.hr_work_location;")

        # hr_job
        vn.train(question="What are the jobs with recruitment currently in progress?", sql="SELECT id, name->>'en_US' AS job_name FROM public.hr_job WHERE no_of_recruitment > 0;")
        vn.train(question="Which jobs have expected employees greater than the current number of employees?",sql="SELECT id, name->>'en_US' AS job_name FROM public.hr_job WHERE expected_employees > no_of_employee;")
        vn.train(question="What are the unique contract types associated with the jobs?", sql="SELECT DISTINCT contract_type_id FROM public.hr_job;")
        vn.train(question="Who is the creator of the 'Chief Technical Officer' job?", sql="SELECT create_uid FROM public.hr_job WHERE name->>'en_US' = 'Chief Technical Officer';")
        vn.train(question="When was the 'Consultant' job last updated?", sql="SELECT write_date FROM public.hr_job WHERE name->>'en_US' = 'Consultant';")
        vn.train(question="What is the description of the job with ID 4?", sql="SELECT description FROM public.hr_job WHERE id=4;")
        vn.train(question="Which jobs are currently active?", sql="SELECT id, name->>'en_US' as job_name FROM public.hr_job WHERE active=true;")
        vn.train(question="Which jobs were created on '2025-09-02'?", sql="SELECT id, name->>'en_US' as job_name FROM public.hr_job WHERE create_date::date = '2025-09-02';")
        vn.train(question="What are the job requirements for 'Chief Executive Officer'?", sql="SELECT requirements FROM public.hr_job WHERE name->>'en_US' = 'Chief Executive Officer';")
        vn.train(question="How many jobs are there in each department?", sql="SELECT department_id, COUNT(id) as job_count FROM public.hr_job GROUP BY department_id;")
        

        ##### JOINS 
        logger.info("Vanna training completed!")
        # Test the connection and data visibility
        try:
            test_sql = vn.generate_sql("List all categories")
            logger.info(f"Test SQL generated: {test_sql}")
        except Exception as e:
            logger.error(f"Test query failed: {e}")

    except Exception as e:
        logger.error(f"Training error: {e}")
else:
    logger.warning("Warning: OPENAI_API_KEY not set, running in demo mode")

# --- Create Flask app ---
app = Flask(__name__, static_folder='static')
# Configure CORS with more permissive settings
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True)
# Add comprehensive CORS headers to all responses
@app.after_request
def after_request(response):
    # accept requests from all domains
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 24 hours
    return response

# Global OPTIONS handler for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response

# --- API Routes ---
@app.route('/')
def index():
    return jsonify({
        "message": "Vanna API is running",
        "status": "ok",
        "endpoints": ["/", "/health", "/query", "/train", "/ui", "/v1/models", "/models", "/v1/chat/completions"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "database": "connected", "llm_data_access": vn.allow_llm_to_see_data if vn else False})

@app.route('/models', methods=['GET', 'OPTIONS'])
def models_redirect():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        return response
    return list_models()

@app.route('/query', methods=['POST', 'OPTIONS'])
def query():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not vn:
        return jsonify({"error": "Vanna not initialized - check OPENAI_API_KEY"}), 500

    try:
        # Ensure allow_llm_to_see_data is True
        vn.allow_llm_to_see_data = True
        sql = vn.generate_sql(question, allow_llm_to_see_data=True)

        if not isinstance(sql, str) or not sql.strip().upper().startswith("SELECT"):
            return jsonify({
                "question": question,
                "sql": sql,
                "error": "LLM did not return valid SQL. Response may be metadata or malformed."
            }), 400

        result = vn.run_sql(sql)

        return jsonify({
            "question": question,
            "sql": sql,
            "result": result.to_dict('records') if hasattr(result, 'to_dict') else str(result)
        })
    except Exception as e:
        return jsonify({
            "question": question,
            "sql": sql if 'sql' in locals() else "N/A",
            "error": f"Execution failed: {str(e)}"
        }), 500

@app.route('/train', methods=['POST'])
def train():
    if not vn:
        return jsonify({"error": "Vanna not initialized"}), 500

    try:
        data = request.get_json()

        if 'ddl' in data:
            vn.train(ddl=data['ddl'])
            return jsonify({"message": "DDL training completed"})
        elif 'sql' in data:
            vn.train(sql=data['sql'])
            return jsonify({"message": "SQL training completed"})
        elif 'documentation' in data:
            vn.train(documentation=data['documentation'])
            return jsonify({"message": "Documentation training completed"})
        else:
            return jsonify({"error": "Provide either 'ddl', 'sql', or 'documentation' in request"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data')
def get_data():
    if not vn:
        return jsonify({"error": "Vanna not initialized"}), 500

    try:
        result = vn.run_sql("SELECT * FROM sales")
        logger.debug(type(result))
        return jsonify(result.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ui')
def ui():
    return redirect(f"http://localhost:{WEBUI_PORT}")

@app.route('/ui/legacy')
def legacy_ui():
    return app.send_static_file('index.html')

# --- OpenAI API Compatible Endpoints ---
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    # remove the default handler
    # add a file handler for this chat only
    file_name = str(datetime.now()).replace(":",";")
    handler2 = logging.FileHandler(os.path.join(".",path_prefix,"debug",file_name+".log"))  # Log file name
    # logger2 = logging.getLogger('logger2')
    # logger2.setLevel(logging.INFO)
    # handler2 = logging.FileHandler(file_handler)
    
    logger.addHandler(handler2)
    logger.removeHandler(handler1)
    logger.setLevel(logging.DEBUG)
    logger.info("<<"*25+"Start of chat"+">>"*25)

    """
    file_name = str(datetime.now()).replace(":",";")
    chat_logger = logging.getLogger('my_chat_logger')
    chat_file_handler = logging.FileHandler(os.path.join(".",path_prefix,"debug",file_name+".log"))  # Log file name
    chat_logger.setLevel(logging.INFO)
    chat_logger.addHandler(chat_file_handler)
    """

    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response
    
    try:
        try:
            data = request.get_json(force=True)  # Force JSON parsing
            logger.info("The received json inside the POST request is as follows")
            logger.info(json.dumps(data,indent=3))
            # chat_logger.info(data.get("mode"))
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return jsonify({"error": {"message": "Invalid JSON format", "type": "invalid_request"}}), 400

        if not data:
            logger.info("No data were passed in the POST request.")
            return jsonify({"error": {"message": "No JSON data received", "type": "invalid_request"}}), 400

        messages = data.get('messages', [])
        # logger.info("messages")
        # for message in messages:
        #    logger.info(json.dumps(message,indent=3))
        model = data.get('model', 'vanna-sql-query-hr')
        # logger.info("model")
        # logger.info(model)
        stream = data.get('stream', False)
        # logger.info("stream")
        # logger.info(stream)    
        
        # Extract user question from messages
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        system_messages = [msg for msg in messages if msg.get('role') != 'user']
        # logger.info(f"Type of messages is {type(messages)}")
        # logger.info("user_messages")
        # for user_message in user_messages:
        #     logger.info(json.dumps(user_message,indent=3))
        # logger.info("System message")
        # for system_message in system_messages:
        #    logger.info(system_message)
        if not user_messages:
            logger.error("No user messages were passed.")
            return jsonify({"error": {"message": "No user message found", "type": "invalid_request"}}), 400
        
        # OpenWEB UI might send more than a user message from the history of the users questions so we take the last one 
        # which is exactly what the user has asked.
        question = user_messages[-1].get('content', '').strip()
        # logger.info("question")
        # here is the models question
        # logger.info(">"*10+question+"<"*10)
        # Generate response
        if vn and question and not question.lower().startswith('hello'):
            try:
                logger.info("Vn is not None, We found a user question, and the question does not start with hello.")
                # Ensure allow_llm_to_see_data is True
                vn.allow_llm_to_see_data = True

                # Try to generate SQL and get results
                sql = vn.generate_sql(question, allow_llm_to_see_data=True)
                logger.info("Vanna has generated the following sql.")
                # here is the generate sql by the model
                logger.info(sql)
                if isinstance(sql, str) and sql.strip().upper().startswith("SELECT"):
                    logger.info("The returned sql is of python's 'str' type, and it starts with select which is safe command to execute.")
                    # here is the models results.
                    result = vn.run_sql(sql)
                    logger.debug("result of the sql generated by vanna is.")
                    # logger.debug(f"The type of the returned results of the run_sql is of type {type(result)}")
                    logger.info(result)
                    # response_text = f"Here's the SQL query I generated for your question:\n\n```sql\n{sql}\n```\n\nResults:\n{result.to_string() if hasattr(result, 'to_string') else str(result)}"
                    response_text = f"Here's the SQL query I generated for your question:\n\n```sql\n{sql}\n```\n\nResults:\n{result.to_markdown(index=False)}"

                else:
                    logger.warning("The sql statement generated by vanna is neither guaranteed to be of type 'str', not to start with select keyword.")
                    logger.warning(sql)
                    response_text = f"I generated this response: {sql}"
            except Exception as e:
                response_text = f"I encountered an error while processing your question: {str(e)}\n\nTip: Make sure your question is related to the data the model has knowledge about."
                logger.error(f"An error was encounter while processing the question: {str(e)}")
                # logger.error(response_text)
        else:
            # Default responses for testing or when Vanna is not available
            if question.lower().startswith('hello') or not question:
                response_text = "Hello! I'm your SQL assistant. I can help you query sales data. Try asking questions like:\n- 'List all products'\n- 'Show me Alice's purchases'\n- 'Which customer spent the most?'\n- 'What's the total revenue?'"
            else:
                response_text = f"I received your question: '{question}'. This is a test response showing that the connection is working."
            logger.error(response_text)
        # Handle streaming vs non-streaming
        if stream:
            logger.debug("Inside stream")
            def generate_stream():
                # Send the response in chunks for streaming
                chunk_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": response_text},
                            "finish_reason": None
                        }
                    ]
                }
                logger.debug("chunk_data")
                logger.debug(json.dumps(chunk_data,indent=3))
                yield f"data: {json.dumps(chunk_data)}\n\n"

                # Send final chunk
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                logger.info("final_chunk")
                logger.info(json.dumps(final_chunk))
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate_stream(),
                          # mimetype='text/plain',
                          mimetype='text/event-stream',
                          headers={
                              'Cache-Control': 'no-cache',
                              'Connection': 'keep-alive',
                              'Access-Control-Allow-Origin': '*'
                          })
        else:
            # Non-streaming response
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(question.split()) if question else 10,
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(question.split()) + len(response_text.split()) if question else 18
                }
            }

            return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        return jsonify({
            "error": {
                "message": f"Internal server error",
                "type": "internal_error"
            }
        }), 500
    finally:
        logger.info("<<"*25+"End of chat"+">>"*25)
        logger.addHandler(handler1)
        logger.removeHandler(handler2)


@app.route('/chat/completions', methods=['POST', 'OPTIONS'])
def legacy_chat_completions():
    if request.method == 'OPTIONS':
        return '', 200
    return chat_completions()

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        return response

    models = [
        # {
        #     "id": "gpt-3.5-turbo",
        #     "object": "model",
        #     "created": int(time.time()),
        #     "owned_by": "openai"
        # },
        {
            "id": "vanna-sql-query-hr",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vanna"
        }
    ]
    return jsonify({"data": models, "object": "list"})

# # Test endpoint to verify API is working
@app.route('/test', methods=['GET', 'OPTIONS'])
def test():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        return response

    return jsonify({
        "message": "API is working",
        "timestamp": int(time.time()),
        "vanna_initialized": vn is not None,
        "allow_llm_to_see_data": vn.allow_llm_to_see_data if vn else False,
        "cors_enabled": True
    })

# Debug endpoint to check CORS
@app.route('/debug/cors', methods=['GET', 'POST', 'OPTIONS'])
def debug_cors():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response

    return jsonify({
        "method": request.method,
        "headers": dict(request.headers),
        "origin": request.headers.get('Origin', 'No Origin'),
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    })
@app.after_request
def after_request(response):
    if request.path.startswith('/v1/') and response.is_json:
        response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response
if __name__ == "__main__":
    print("Starting services...")

    print(f"\n=== Setup Instructions ===")
    print(f"1. Start this Flask API (already running)")
    print(f"2. In Open WebUI settings:")
    print(f"   - Go to Settings > Connections")
    print(f"   - Set OpenAI API Base URL to: http://localhost:{API_PORT}")
    print(f"   - Set OpenAI API Key to: dummy-key (or any value)")
    print(f"   - Save settings")
    print(f"3. Select model 'vanna-sql-query-hr' in the chat interface")
    print(f"4. Test with: 'Hello' or 'List all products'")
    print(f"\n=== URLs ===")
    print(f"- API: http://localhost:{API_PORT}")
    print(f"- Test endpoint: http://localhost:{API_PORT}/test")
    print(f"- Models: http://localhost:{API_PORT}/v1/models")
    print(f"==========================")

    # Run Flask app
    # app.run(host="0.0.0.0", port=API_PORT, debug=True)
