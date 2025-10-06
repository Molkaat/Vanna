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
API_PORT = int(os.getenv("API_PORT", "8787"))
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

import logging
# Create a logger
logger = logging.getLogger('my_logger')
# Create a file handler
file_handler = logging.FileHandler('app.log')  # Log file name
file_handler.setLevel(logging.DEBUG)

# Create a log formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Line %(lineno)d - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


# --- Reset ChromaDB if needed ---
def reset_chromadb():
    chroma_path = "chroma.sqlite3"
    if os.path.exists(chroma_path):
        try:
            backup_path = f"chroma.sqlite3.bak.{int(time.time())}"
            shutil.move(chroma_path, backup_path)
            print(f"Backed up existing ChromaDB to {backup_path}")
        except Exception as e:
            print(f"Error backing up ChromaDB: {e}")
            try:
                os.remove(chroma_path)
                print(f"Deleted existing ChromaDB file")
            except Exception as e:
                print(f"Error deleting ChromaDB file: {e}")
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
        logger.info("get_similar_question_sql")
        logger.info(anchor)
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
        logger.info("get_related_ddl")
        logger.info(anchor)
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
        logger.info("get_related_documentation")
        logger.info(anchor)
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
        logger.info("submit_prompt")
        logger.info(anchor)
        json.dumps(results,indent=3)
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
        "n_results":2,
        "temperature":LLM_TEMPERATURE
    })

    logger.debug("LLM temperature is ")

    # Set the allow_llm_to_see_data attribute directly
    vn.allow_llm_to_see_data = True

    # db_path = create_database()
    # vn.connect_to_sqlite(db_path)
    vn.connect_to_postgres(host='3.29.246.202', dbname='odoo', user='odoo18', password='odoo18', port='5432')
    print("Database connected!")

    ddls = [
        """CREATE TABLE public.account_account(    id integer NOT NULL DEFAULT nextval('account_account_id_seq'::regclass),    message_main_attachment_id integer,    currency_id integer,    company_id integer NOT NULL,    group_id integer,    root_id integer,    create_uid integer,    write_uid integer,    name character varying  NOT NULL,    code character varying(64)  NOT NULL,    account_type character varying  NOT NULL,    internal_group character varying ,    note text ,    deprecated boolean,    include_initial_balance boolean,    reconcile boolean,    is_off_balance boolean,    non_trade boolean,    create_date timestamp without time zone,    write_date timestamp without time zone,    CONSTRAINT account_account_pkey PRIMARY KEY (id),    CONSTRAINT account_account_code_company_uniq UNIQUE (code, company_id),    CONSTRAINT account_account_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.res_company (id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE RESTRICT,    CONSTRAINT account_account_create_uid_fkey FOREIGN KEY (create_uid) REFERENCES public.res_users (id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE SET NULL,    CONSTRAINT account_account_currency_id_fkey FOREIGN KEY (currency_id) REFERENCES public.res_currency (id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE SET NULL,    CONSTRAINT account_account_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.account_group (id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE SET NULL,    CONSTRAINT account_account_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id) REFERENCES public.ir_attachment (id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE SET NULL,    CONSTRAINT account_account_write_uid_fkey FOREIGN KEY (write_uid) REFERENCES public.res_users (id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE SET NULL);""",
        """
        CREATE TABLE public.hr_employee
(
    id integer NOT NULL DEFAULT nextval('hr_employee_id_seq'::regclass),
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
        CREATE TABLE public.sale_order
        (
    id integer NOT NULL DEFAULT nextval('sale_order_id_seq'::regclass),
    campaign_id integer,
    source_id integer,
    medium_id integer,
    message_main_attachment_id integer,
    company_id integer NOT NULL,
    partner_id integer NOT NULL,
    partner_invoice_id integer NOT NULL,
    partner_shipping_id integer NOT NULL,
    fiscal_position_id integer,
    payment_term_id integer,
    pricelist_id integer NOT NULL,
    currency_id integer,
    user_id integer,
    team_id integer,
    analytic_account_id integer,
    create_uid integer,
    write_uid integer,
    access_token character varying ,
    name character varying  NOT NULL,
    state character varying ,
    client_order_ref character varying ,
    origin character varying ,
    reference character varying ,
    signed_by character varying ,
    invoice_status character varying ,
    validity_date date,
    note text ,
    currency_rate numeric,
    amount_untaxed numeric,
    amount_tax numeric,
    amount_total numeric,
    require_signature boolean,
    require_payment boolean,
    create_date timestamp without time zone,
    commitment_date timestamp without time zone,
    date_order timestamp without time zone NOT NULL,
    signed_on timestamp without time zone,
    write_date timestamp without time zone,
    opportunity_id integer,
    incoterm integer,
    warehouse_id integer NOT NULL,
    procurement_group_id integer,
    incoterm_location character varying ,
    picking_policy character varying  NOT NULL,
    delivery_status character varying ,
    effective_date timestamp without time zone,
    CONSTRAINT sale_order_pkey PRIMARY KEY (id),
    CONSTRAINT sale_order_analytic_account_id_fkey FOREIGN KEY (analytic_account_id)
        REFERENCES public.account_analytic_account (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_campaign_id_fkey FOREIGN KEY (campaign_id)
        REFERENCES public.utm_campaign (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_currency_id_fkey FOREIGN KEY (currency_id)
        REFERENCES public.res_currency (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_fiscal_position_id_fkey FOREIGN KEY (fiscal_position_id)
        REFERENCES public.account_fiscal_position (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_incoterm_fkey FOREIGN KEY (incoterm)
        REFERENCES public.account_incoterms (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_medium_id_fkey FOREIGN KEY (medium_id)
        REFERENCES public.utm_medium (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id)
        REFERENCES public.ir_attachment (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_opportunity_id_fkey FOREIGN KEY (opportunity_id)
        REFERENCES public.crm_lead (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_partner_invoice_id_fkey FOREIGN KEY (partner_invoice_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_partner_shipping_id_fkey FOREIGN KEY (partner_shipping_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_payment_term_id_fkey FOREIGN KEY (payment_term_id)
        REFERENCES public.account_payment_term (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_pricelist_id_fkey FOREIGN KEY (pricelist_id)
        REFERENCES public.product_pricelist (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_procurement_group_id_fkey FOREIGN KEY (procurement_group_id)
        REFERENCES public.procurement_group (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_source_id_fkey FOREIGN KEY (source_id)
        REFERENCES public.utm_source (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_team_id_fkey FOREIGN KEY (team_id)
        REFERENCES public.crm_team (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_warehouse_id_fkey FOREIGN KEY (warehouse_id)
        REFERENCES public.stock_warehouse (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_date_order_conditional_required CHECK ((state::text = ANY (ARRAY['sale'::character varying, 'done'::character varying]::text[])) AND date_order IS NOT NULL OR (state::text <> ALL (ARRAY['sale'::character varying, 'done'::character varying]::text[])))
)
        """,
        """
        CREATE TABLE public.sale_order_line
(
    id integer NOT NULL DEFAULT nextval('sale_order_line_id_seq'::regclass),
    order_id integer NOT NULL,
    sequence integer,
    company_id integer,
    currency_id integer,
    order_partner_id integer,
    salesman_id integer,
    product_id integer,
    product_uom integer,
    product_packaging_id integer,
    create_uid integer,
    write_uid integer,
    state character varying ,
    display_type character varying ,
    qty_delivered_method character varying ,
    invoice_status character varying ,
    analytic_distribution jsonb,
    name text  NOT NULL,
    product_uom_qty numeric NOT NULL,
    price_unit numeric NOT NULL,
    discount numeric,
    price_reduce numeric,
    price_subtotal numeric,
    price_total numeric,
    price_reduce_taxexcl numeric,
    price_reduce_taxinc numeric,
    qty_delivered numeric,
    qty_invoiced numeric,
    qty_to_invoice numeric,
    untaxed_amount_invoiced numeric,
    untaxed_amount_to_invoice numeric,
    is_downpayment boolean,
    is_expense boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    price_tax double precision,
    product_packaging_qty double precision,
    customer_lead double precision NOT NULL,
    route_id integer,
    CONSTRAINT sale_order_line_pkey PRIMARY KEY (id),
    CONSTRAINT sale_order_line_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_currency_id_fkey FOREIGN KEY (currency_id)
        REFERENCES public.res_currency (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_order_id_fkey FOREIGN KEY (order_id)
        REFERENCES public.sale_order (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_order_partner_id_fkey FOREIGN KEY (order_partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_product_id_fkey FOREIGN KEY (product_id)
        REFERENCES public.product_product (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_product_packaging_id_fkey FOREIGN KEY (product_packaging_id)
        REFERENCES public.product_packaging (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_product_uom_fkey FOREIGN KEY (product_uom)
        REFERENCES public.uom_uom (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_route_id_fkey FOREIGN KEY (route_id)
        REFERENCES public.stock_route (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_salesman_id_fkey FOREIGN KEY (salesman_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT sale_order_line_accountable_required_fields CHECK (display_type IS NOT NULL OR product_id IS NOT NULL AND product_uom IS NOT NULL),
    CONSTRAINT sale_order_line_non_accountable_null_fields CHECK (display_type IS NULL OR product_id IS NULL AND price_unit = 0::numeric AND product_uom_qty = 0::numeric AND product_uom IS NULL AND customer_lead = 0::double precision)
)
        """,
        """
        CREATE TABLE public.product_template
(
    id integer NOT NULL DEFAULT nextval('product_template_id_seq'::regclass),
    message_main_attachment_id integer,
    sequence integer,
    categ_id integer NOT NULL,
    uom_id integer NOT NULL,
    uom_po_id integer NOT NULL,
    company_id integer,
    color integer,
    create_uid integer,
    write_uid integer,
    detailed_type character varying  NOT NULL,
    type character varying ,
    default_code character varying ,
    priority character varying ,
    name jsonb NOT NULL,
    description jsonb,
    description_purchase jsonb,
    description_sale jsonb,
    list_price numeric,
    volume numeric,
    weight numeric,
    sale_ok boolean,
    purchase_ok boolean,
    active boolean,
    can_image_1024_be_zoomed boolean,
    has_configurable_attributes boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    tracking character varying  NOT NULL,
    description_picking jsonb,
    description_pickingout jsonb,
    description_pickingin jsonb,
    sale_delay double precision,
    purchase_method character varying ,
    purchase_line_warn character varying  NOT NULL,
    purchase_line_warn_msg text ,
    pos_categ_id integer,
    available_in_pos boolean,
    to_weight boolean,
    service_type character varying ,
    sale_line_warn character varying  NOT NULL,
    expense_policy character varying ,
    invoice_policy character varying ,
    sale_line_warn_msg text ,
    CONSTRAINT product_template_pkey PRIMARY KEY (id),
    CONSTRAINT product_template_categ_id_fkey FOREIGN KEY (categ_id)
        REFERENCES public.product_category (id) MATCH SIMPLE
,
    CONSTRAINT product_template_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT product_template_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT product_template_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id)
        REFERENCES public.ir_attachment (id) MATCH SIMPLE
,
    CONSTRAINT product_template_pos_categ_id_fkey FOREIGN KEY (pos_categ_id)
        REFERENCES public.pos_category (id) MATCH SIMPLE
,
    CONSTRAINT product_template_uom_id_fkey FOREIGN KEY (uom_id)
        REFERENCES public.uom_uom (id) MATCH SIMPLE
,
    CONSTRAINT product_template_uom_po_id_fkey FOREIGN KEY (uom_po_id)
        REFERENCES public.uom_uom (id) MATCH SIMPLE
,
    CONSTRAINT product_template_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
)
        """,
        """
        CREATE TABLE public.stock_quant
(
    id integer NOT NULL DEFAULT nextval('stock_quant_id_seq'::regclass),
    product_id integer NOT NULL,
    company_id integer,
    location_id integer NOT NULL,
    storage_category_id integer,
    lot_id integer,
    package_id integer,
    owner_id integer,
    user_id integer,
    create_uid integer,
    write_uid integer,
    inventory_date date,
    quantity numeric,
    reserved_quantity numeric NOT NULL,
    inventory_quantity numeric,
    inventory_diff_quantity numeric,
    inventory_quantity_set boolean,
    in_date timestamp without time zone NOT NULL,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    accounting_date date,
    CONSTRAINT stock_quant_pkey PRIMARY KEY (id),
    CONSTRAINT stock_quant_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_location_id_fkey FOREIGN KEY (location_id)
        REFERENCES public.stock_location (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_lot_id_fkey FOREIGN KEY (lot_id)
        REFERENCES public.stock_lot (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_owner_id_fkey FOREIGN KEY (owner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_package_id_fkey FOREIGN KEY (package_id)
        REFERENCES public.stock_quant_package (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_product_id_fkey FOREIGN KEY (product_id)
        REFERENCES public.product_product (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_storage_category_id_fkey FOREIGN KEY (storage_category_id)
        REFERENCES public.stock_storage_category (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT stock_quant_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
)
        """,
        """
        CREATE TABLE public.stock_move
(
    id integer NOT NULL DEFAULT nextval('stock_move_id_seq'::regclass),
    sequence integer,
    company_id integer NOT NULL,
    product_id integer NOT NULL,
    product_uom integer NOT NULL,
    location_id integer NOT NULL,
    location_dest_id integer NOT NULL,
    partner_id integer,
    picking_id integer,
    group_id integer,
    rule_id integer,
    picking_type_id integer,
    origin_returned_move_id integer,
    restrict_partner_id integer,
    warehouse_id integer,
    package_level_id integer,
    next_serial_count integer,
    orderpoint_id integer,
    product_packaging_id integer,
    create_uid integer,
    write_uid integer,
    name character varying  NOT NULL,
    priority character varying ,
    state character varying ,
    origin character varying ,
    procure_method character varying  NOT NULL,
    reference character varying ,
    next_serial character varying ,
    reservation_date date,
    description_picking text ,
    product_qty numeric,
    product_uom_qty numeric NOT NULL,
    quantity_done numeric,
    scrapped boolean,
    propagate_cancel boolean,
    is_inventory boolean,
    additional boolean,
    date timestamp without time zone NOT NULL,
    date_deadline timestamp without time zone,
    delay_alert_date timestamp without time zone,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    price_unit double precision,
    analytic_account_line_id integer,
    to_refund boolean,
    purchase_line_id integer,
    created_purchase_line_id integer,
    sale_line_id integer,
    CONSTRAINT stock_move_pkey PRIMARY KEY (id),
    CONSTRAINT stock_move_analytic_account_line_id_fkey FOREIGN KEY (analytic_account_line_id)
        REFERENCES public.account_analytic_line (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_created_purchase_line_id_fkey FOREIGN KEY (created_purchase_line_id)
        REFERENCES public.purchase_order_line (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_group_id_fkey FOREIGN KEY (group_id)
        REFERENCES public.procurement_group (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_location_dest_id_fkey FOREIGN KEY (location_dest_id)
        REFERENCES public.stock_location (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_location_id_fkey FOREIGN KEY (location_id)
        REFERENCES public.stock_location (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_orderpoint_id_fkey FOREIGN KEY (orderpoint_id)
        REFERENCES public.stock_warehouse_orderpoint (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_origin_returned_move_id_fkey FOREIGN KEY (origin_returned_move_id)
        REFERENCES public.stock_move (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_package_level_id_fkey FOREIGN KEY (package_level_id)
        REFERENCES public.stock_package_level (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_picking_id_fkey FOREIGN KEY (picking_id)
        REFERENCES public.stock_picking (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_picking_type_id_fkey FOREIGN KEY (picking_type_id)
        REFERENCES public.stock_picking_type (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_product_id_fkey FOREIGN KEY (product_id)
        REFERENCES public.product_product (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_product_packaging_id_fkey FOREIGN KEY (product_packaging_id)
        REFERENCES public.product_packaging (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_product_uom_fkey FOREIGN KEY (product_uom)
        REFERENCES public.uom_uom (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_purchase_line_id_fkey FOREIGN KEY (purchase_line_id)
        REFERENCES public.purchase_order_line (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_restrict_partner_id_fkey FOREIGN KEY (restrict_partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_rule_id_fkey FOREIGN KEY (rule_id)
        REFERENCES public.stock_rule (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_sale_line_id_fkey FOREIGN KEY (sale_line_id)
        REFERENCES public.sale_order_line (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_warehouse_id_fkey FOREIGN KEY (warehouse_id)
        REFERENCES public.stock_warehouse (id) MATCH SIMPLE
,
    CONSTRAINT stock_move_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
)
        """,
        """
        CREATE TABLE public.res_users
(
    id integer NOT NULL DEFAULT nextval('res_users_id_seq'::regclass),
    company_id integer NOT NULL,
    partner_id integer NOT NULL,
    active boolean DEFAULT true,
    create_date timestamp without time zone,
    login character varying  NOT NULL,
    password character varying ,
    action_id integer,
    create_uid integer,
    write_uid integer,
    signature text ,
    share boolean,
    write_date timestamp without time zone,
    totp_secret character varying ,
    notification_type character varying  NOT NULL,
    odoobot_state character varying ,
    odoobot_failed boolean,
    sale_team_id integer,
    target_sales_won integer,
    target_sales_done integer,
    website_id integer,
    target_sales_invoiced integer,
    CONSTRAINT res_users_pkey PRIMARY KEY (id),
    CONSTRAINT res_users_login_key UNIQUE (login, website_id),
    CONSTRAINT res_users_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT res_users_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT res_users_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT res_users_sale_team_id_fkey FOREIGN KEY (sale_team_id)
        REFERENCES public.crm_team (id) MATCH SIMPLE
,
    CONSTRAINT res_users_website_id_fkey FOREIGN KEY (website_id)
        REFERENCES public.website (id) MATCH SIMPLE
,
    CONSTRAINT res_users_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT res_users_notification_type CHECK (notification_type::text = 'email'::text OR NOT share)
)
        """,
        """
        CREATE TABLE public.project_project
(
    id integer NOT NULL DEFAULT nextval('project_project_id_seq'::regclass),
    message_main_attachment_id integer,
    alias_id integer NOT NULL,
    sequence integer,
    partner_id integer,
    company_id integer NOT NULL,
    analytic_account_id integer,
    color integer,
    user_id integer,
    stage_id integer,
    last_update_id integer,
    create_uid integer,
    write_uid integer,
    access_token character varying ,
    partner_email character varying ,
    partner_phone character varying ,
    privacy_visibility character varying  NOT NULL,
    rating_status character varying  NOT NULL,
    rating_status_period character varying  NOT NULL,
    last_update_status character varying  NOT NULL,
    date_start date,
    date date,
    name jsonb NOT NULL,
    label_tasks jsonb,
    task_properties_definition jsonb,
    description text ,
    active boolean,
    allow_subtasks boolean,
    allow_recurring_tasks boolean,
    allow_task_dependencies boolean,
    allow_milestones boolean,
    rating_active boolean,
    rating_request_deadline timestamp without time zone,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT project_project_pkey PRIMARY KEY (id),
    CONSTRAINT project_project_alias_id_fkey FOREIGN KEY (alias_id)
        REFERENCES public.mail_alias (id) MATCH SIMPLE
,
    CONSTRAINT project_project_analytic_account_id_fkey FOREIGN KEY (analytic_account_id)
        REFERENCES public.account_analytic_account (id) MATCH SIMPLE
,
    CONSTRAINT project_project_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT project_project_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT project_project_last_update_id_fkey FOREIGN KEY (last_update_id)
        REFERENCES public.project_update (id) MATCH SIMPLE
,
    CONSTRAINT project_project_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id)
        REFERENCES public.ir_attachment (id) MATCH SIMPLE
,
    CONSTRAINT project_project_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT project_project_stage_id_fkey FOREIGN KEY (stage_id)
        REFERENCES public.project_project_stage (id) MATCH SIMPLE
,
    CONSTRAINT project_project_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT project_project_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT project_project_project_date_greater CHECK (date >= date_start)
)
        """,
        """
        
CREATE TABLE public.crm_lead
(
    id integer NOT NULL DEFAULT nextval('crm_lead_id_seq'::regclass),
    campaign_id integer,
    source_id integer,
    medium_id integer,
    message_main_attachment_id integer,
    message_bounce integer,
    user_id integer,
    team_id integer,
    company_id integer,
    stage_id integer,
    color integer,
    recurring_plan integer,
    partner_id integer,
    title integer,
    lang_id integer,
    state_id integer,
    country_id integer,
    lost_reason_id integer,
    create_uid integer,
    write_uid integer,
    phone_sanitized character varying ,
    email_normalized character varying ,
    email_cc character varying ,
    name character varying  NOT NULL,
    referred character varying ,
    type character varying  NOT NULL,
    priority character varying ,
    contact_name character varying ,
    partner_name character varying ,
    function character varying ,
    email_from character varying ,
    phone character varying ,
    mobile character varying ,
    phone_state character varying ,
    email_state character varying ,
    website character varying ,
    street character varying ,
    street2 character varying ,
    zip character varying ,
    city character varying ,
    date_deadline date,
    lead_properties jsonb,
    description text ,
    expected_revenue numeric,
    prorated_revenue numeric,
    recurring_revenue numeric,
    recurring_revenue_monthly numeric,
    recurring_revenue_monthly_prorated numeric,
    active boolean,
    date_closed timestamp without time zone,
    date_action_last timestamp without time zone,
    date_open timestamp without time zone,
    date_last_stage_update timestamp without time zone,
    date_conversion timestamp without time zone,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    day_open double precision,
    day_close double precision,
    probability double precision,
    automated_probability double precision,
    reveal_id character varying ,
    iap_enrich_done boolean,
    lead_mining_request_id integer,
    CONSTRAINT crm_lead_pkey PRIMARY KEY (id),
    CONSTRAINT crm_lead_campaign_id_fkey FOREIGN KEY (campaign_id)
        REFERENCES public.utm_campaign (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_country_id_fkey FOREIGN KEY (country_id)
        REFERENCES public.res_country (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_lang_id_fkey FOREIGN KEY (lang_id)
        REFERENCES public.res_lang (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_lead_mining_request_id_fkey FOREIGN KEY (lead_mining_request_id)
        REFERENCES public.crm_iap_lead_mining_request (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_lost_reason_id_fkey FOREIGN KEY (lost_reason_id)
        REFERENCES public.crm_lost_reason (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_medium_id_fkey FOREIGN KEY (medium_id)
        REFERENCES public.utm_medium (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_message_main_attachment_id_fkey FOREIGN KEY (message_main_attachment_id)
        REFERENCES public.ir_attachment (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_recurring_plan_fkey FOREIGN KEY (recurring_plan)
        REFERENCES public.crm_recurring_plan (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_source_id_fkey FOREIGN KEY (source_id)
        REFERENCES public.utm_source (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_stage_id_fkey FOREIGN KEY (stage_id)
        REFERENCES public.crm_stage (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_state_id_fkey FOREIGN KEY (state_id)
        REFERENCES public.res_country_state (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_team_id_fkey FOREIGN KEY (team_id)
        REFERENCES public.crm_team (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_title_fkey FOREIGN KEY (title)
        REFERENCES public.res_partner_title (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT crm_lead_check_probability CHECK (probability >= 0::double precision AND probability <= 100::double precision)
)
        """,
        """
CREATE TABLE public.purchase_order
(
    id integer NOT NULL DEFAULT nextval('purchase_order_id_seq'::regclass),
    partner_id integer NOT NULL,
    dest_address_id integer,
    currency_id integer NOT NULL,
    invoice_count integer,
    fiscal_position_id integer,
    payment_term_id integer,
    incoterm_id integer,
    user_id integer,
    company_id integer NOT NULL,
    create_uid integer,
    write_uid integer,
    access_token character varying ,
    name character varying  NOT NULL,
    priority character varying ,
    origin character varying ,
    partner_ref character varying ,
    state character varying ,
    invoice_status character varying ,
    notes text ,
    amount_untaxed numeric,
    amount_tax numeric,
    amount_total numeric,
    amount_total_cc numeric,
    currency_rate numeric,
    mail_reminder_confirmed boolean,
    mail_reception_confirmed boolean,
    mail_reception_declined boolean,
    date_order timestamp without time zone NOT NULL,
    date_approve timestamp without time zone,
    date_planned timestamp without time zone,
    date_calendar_start timestamp without time zone,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    picking_type_id integer NOT NULL,
    group_id integer,
    incoterm_location character varying ,
    receipt_status character varying ,
    effective_date timestamp without time zone,
    CONSTRAINT purchase_order_pkey PRIMARY KEY (id),
    CONSTRAINT purchase_order_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_currency_id_fkey FOREIGN KEY (currency_id)
        REFERENCES public.res_currency (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_dest_address_id_fkey FOREIGN KEY (dest_address_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_fiscal_position_id_fkey FOREIGN KEY (fiscal_position_id)
        REFERENCES public.account_fiscal_position (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_group_id_fkey FOREIGN KEY (group_id)
        REFERENCES public.procurement_group (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_incoterm_id_fkey FOREIGN KEY (incoterm_id)
        REFERENCES public.account_incoterms (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_payment_term_id_fkey FOREIGN KEY (payment_term_id)
        REFERENCES public.account_payment_term (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_picking_type_id_fkey FOREIGN KEY (picking_type_id)
        REFERENCES public.stock_picking_type (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.res_users (id) MATCH SIMPLE
,
    CONSTRAINT purchase_order_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
)
        """,
        """
        CREATE TABLE public.purchase_order_line
(
    id integer NOT NULL DEFAULT nextval('purchase_order_line_id_seq'::regclass),
    sequence integer,
    product_uom integer,
    product_id integer,
    order_id integer NOT NULL,
    company_id integer,
    partner_id integer,
    currency_id integer,
    product_packaging_id integer,
    create_uid integer,
    write_uid integer,
    state character varying ,
    qty_received_method character varying ,
    display_type character varying ,
    analytic_distribution jsonb,
    name text  NOT NULL,
    product_qty numeric NOT NULL,
    discount numeric,
    price_unit numeric NOT NULL,
    price_subtotal numeric,
    price_total numeric,
    qty_invoiced numeric,
    qty_received numeric,
    qty_received_manual numeric,
    qty_to_invoice numeric,
    is_downpayment boolean,
    date_planned timestamp without time zone,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    product_uom_qty double precision,
    price_tax double precision,
    product_packaging_qty double precision,
    orderpoint_id integer,
    location_final_id integer,
    group_id integer,
    product_description_variants character varying ,
    propagate_cancel boolean,
    sale_order_id integer,
    sale_line_id integer,
    CONSTRAINT purchase_order_line_pkey PRIMARY KEY (id),
    CONSTRAINT purchase_order_line_company_id_fkey FOREIGN KEY (company_id)
        REFERENCES public.res_company (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_currency_id_fkey FOREIGN KEY (currency_id)
        REFERENCES public.res_currency (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_group_id_fkey FOREIGN KEY (group_id)
        REFERENCES public.procurement_group (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_location_final_id_fkey FOREIGN KEY (location_final_id)
        REFERENCES public.stock_location (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_order_id_fkey FOREIGN KEY (order_id)
        REFERENCES public.purchase_order (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_orderpoint_id_fkey FOREIGN KEY (orderpoint_id)
        REFERENCES public.stock_warehouse_orderpoint (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_partner_id_fkey FOREIGN KEY (partner_id)
        REFERENCES public.res_partner (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_product_id_fkey FOREIGN KEY (product_id)
        REFERENCES public.product_product (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_product_packaging_id_fkey FOREIGN KEY (product_packaging_id)
        REFERENCES public.product_packaging (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_product_uom_fkey FOREIGN KEY (product_uom)
        REFERENCES public.uom_uom (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_sale_line_id_fkey FOREIGN KEY (sale_line_id)
        REFERENCES public.sale_order_line (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_sale_order_id_fkey FOREIGN KEY (sale_order_id)
        REFERENCES public.sale_order (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE,
    CONSTRAINT purchase_order_line_accountable_required_fields CHECK (display_type IS NOT NULL OR is_downpayment OR product_id IS NOT NULL AND product_uom IS NOT NULL AND date_planned IS NOT NULL),
    CONSTRAINT purchase_order_line_non_accountable_null_fields CHECK (display_type IS NULL OR product_id IS NULL AND price_unit = 0::numeric AND product_uom_qty = 0::double precision AND product_uom IS NULL AND date_planned IS NULL)
)s
        """ , 
        """
        
    CREATE TABLE public.crm_stage
    (
    id integer NOT NULL DEFAULT nextval('crm_stage_id_seq'::regclass),
    sequence integer,
    team_id integer,
    create_uid integer,
    write_uid integer,
    name jsonb NOT NULL,
    requirements text ,
    is_won boolean,
    fold boolean,
    create_date timestamp without time zone,
    write_date timestamp without time zone,
    CONSTRAINT crm_stage_pkey PRIMARY KEY (id),
    CONSTRAINT crm_stage_create_uid_fkey FOREIGN KEY (create_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
    ,
    CONSTRAINT crm_stage_team_id_fkey FOREIGN KEY (team_id)
        REFERENCES public.crm_team (id) MATCH SIMPLE
    ,
    CONSTRAINT crm_stage_write_uid_fkey FOREIGN KEY (write_uid)
        REFERENCES public.res_users (id) MATCH SIMPLE
    )
    """
    ]
    for ddl in ddls:
        try:
            vn.train(ddl=ddl)
        except Exception as e:
            print(e)
            print(f"Error training the ddls specifically ddl {ddl}.")
            sys.exit(1)
    # Train Vanna with your database schema and sample data
    try:
        vn.train(question="What are the available names in the accounts table",sql="SELECT name FROM public.account_account; ")
        vn.train(question="What are the hr departments, show some of stored hr departments",sql="select name from hr_department")
        vn.train(question="What is meant by the R&D USA department",sql="select complete_name from hr_department where name = 'R&D USA';")
        vn.train(question="Is the R&D USA department working?, what is the status of the R&D USA department?",sql="select active from hr_department where lower(name) = lower('R&D USA') or lower(complete_name) = lower('R&D USA');")
        vn.train(question="How many sales orders are there?, What is th count of the sales orders?",sql="select count(*) from sale_order where state = 'sale';")
        vn.train(question="How many draft orders are there?, What is th count of the incomplete orders?",sql="select count(*) from sale_order where state = 'draft';")
        vn.train(question="What is our biggest order?, what is the order that generated the most revenue?",sql="select id , amount_total from sale_order order by amount_total desc limit 1")
        vn.train(question="What are the product templates that exist in the product table and does not exist in the product template table",sql="SELECT product_tmpl_id , pt.id from public.product_product pp right outer join public.product_template pt on pp.product_tmpl_id = pt.id where pt.id is null")
        vn.train(question="What is the latest operation on our stock?",sql="select * from public.stock_move order by write_date desc limit 1;")
        vn.train(question="show the distribution of stock operations",sql="select state , count(*) as stateCount from public.stock_move group by state")
        vn.train(question="How many active users are there?",sql="SELECT COUNT(*) FROM public.res_users where active = 'true';")
        vn.train(question="Show the distribution of the notification methods or notification type of each user, what is the notification type of different users",sql="select notification_type , count(*) as countOfNotificationMethod from public.res_users group by notification_type;")
        vn.train(question="Show the user from recent user to the older user, show a sample of our users and show them from latest one to older ones",sql="select * from public.res_users order by write_date desc;")
        vn.train(question="Show a sample of our projects, what are our available projects",sql="SELECT name#>'{en_US}'  FROM public.project_project;")
        vn.train(question="Show all the active projects, is there any active projects, what are the ongoing projects",sql="select * from public.project_project where active = 'true';")
        vn.train(question="How many leads were created in the last 30 days?",sql="SELECT COUNT(*) AS leads_last_30_days FROM crm_lead WHERE create_date >= CURRENT_DATE - INTERVAL '30 days';");        
        vn.train(question="What is the total expected revenue from open leads?", sql="SELECT SUM(expected_revenue) AS total_expected_revenue FROM crm_lead WHERE date_closed IS NULL;");  
        # HAS TO BE TESTED AND CHECKED
        vn.train(question="How many leads were closed successfully (won) vs lost?", sql="SELECT CASE WHEN probability = 100 THEN 'Won' WHEN probability = 0 THEN 'Lost' ELSE 'In Progress' END AS lead_status, COUNT(*) AS lead_count FROM crm_lead GROUP BY lead_status;");
        vn.train(question="What is the average time (in days) to close a lead?", sql="SELECT AVG(day_close) AS avg_days_to_close FROM crm_lead WHERE day_close IS NOT NULL;");
        vn.train(question="Which salespersons have the highest number of leads?", sql="SELECT user_id, COUNT(*) AS lead_count FROM crm_lead GROUP BY user_id ORDER BY lead_count DESC LIMIT 5;");
        vn.train(question="What is the total recurring revenue by country across all the crm leads?", sql="SELECT country_id, SUM(recurring_revenue) AS total_recurring_revenue FROM crm_lead GROUP BY country_id ORDER BY total_recurring_revenue DESC;");
        vn.train(question="What is the conversion rate of leads (won/total)?", sql="SELECT ROUND((SUM(CASE WHEN probability = 100 THEN 1 ELSE 0 END)::decimal / COUNT(*)) * 100, 2) AS conversion_rate_percent FROM crm_lead;");
        vn.train(question="How many leads are currently active and past their deadline?", sql="SELECT COUNT(*) AS overdue_active_leads FROM crm_lead WHERE active = true AND date_deadline IS NOT NULL AND date_deadline < CURRENT_DATE;");
        vn.train(question="What are the top 5 cities with the most leads?", sql="SELECT city, COUNT(*) AS lead_count FROM crm_lead GROUP BY city ORDER BY lead_count DESC LIMIT 5;");
        vn.train(question="List all leads with sanitized phone numbers and valid emails.", sql="SELECT id, name, phone_sanitized, email_normalized FROM crm_lead WHERE phone_state = 'correct' AND email_state = 'correct';");
        # purchase order
        vn.train(question="What is the total amount for purchase order P00001?", sql="SELECT amount_total FROM purchase_order WHERE name = 'P00001';")
        vn.train(question="How many purchase orders are currently in draft state?", sql="SELECT COUNT(*) FROM purchase_order WHERE state = 'draft';")
        vn.train(question="What is the average total amount of all purchase orders?", sql="SELECT AVG(amount_total) FROM purchase_order;")
        vn.train(question="Which purchase order has the highest total amount?", sql="SELECT name, amount_total FROM purchase_order ORDER BY amount_total DESC LIMIT 1;")
        vn.train(question="List all purchase orders with total amount greater than 5000.", sql="SELECT name, amount_total FROM purchase_order WHERE amount_total > 5000;")
        vn.train(question="How many purchase orders have a tax amount greater than 0?", sql="SELECT COUNT(*) FROM purchase_order WHERE amount_tax > 0;")
        vn.train(question="What is the total untaxed amount across all purchase orders?", sql="SELECT SUM(amount_untaxed) FROM purchase_order;")
        vn.train(question="What is the invoice status of purchase order P00002?", sql="SELECT invoice_status FROM purchase_order WHERE name = 'P00002';")
        vn.train(question="Which purchase orders were approved after September 2, 2025?", sql="SELECT name, date_approve FROM purchase_order WHERE date_approve > '2025-09-02';")
        vn.train(question="Show all purchase orders created by user ID 1.", sql="SELECT name FROM purchase_order WHERE create_uid = 1;")
        vn.train(question="Which purchase orders do not have a destination address?", sql="SELECT name FROM purchase_order WHERE dest_address_id IS NULL;")
        vn.train(question="How many purchase orders were created on September 1, 2025?", sql="SELECT COUNT(*) FROM purchase_order WHERE DATE(date_order) = '2025-09-01';")
        vn.train(question="List all distinct states of purchase orders.", sql="SELECT DISTINCT state FROM purchase_order;")
        vn.train(question="Which purchase orders have a non-null access token?", sql="SELECT name, access_token FROM purchase_order WHERE access_token IS NOT NULL;")
        vn.train(question="List purchase orders with currency rate not equal to 1.0.", sql="SELECT name, currency_rate FROM purchase_order WHERE currency_rate <> 1.0;")
        vn.train(question="How many purchase orders were written by user ID 1?", sql="SELECT COUNT(*) FROM purchase_order WHERE write_uid = 1;")
        vn.train(question="Get total number of purchase orders per partner.", sql="SELECT partner_id, COUNT(*) FROM purchase_order GROUP BY partner_id;")
        vn.train(question="Which purchase orders have invoice count greater than 0?", sql="SELECT name, invoice_count FROM purchase_order WHERE invoice_count > 0;")
        vn.train(question="List all purchase orders in the sent state.", sql="SELECT name FROM purchase_order WHERE state = 'sent';")
        vn.train(question="Which purchase orders were planned after the order date?", sql="SELECT name FROM purchase_order WHERE date_planned > date_order;")
        vn.train(question="List all purchase orders with priority 0.", sql="SELECT name FROM purchase_order WHERE priority = '0';")
        vn.train(question="Get the total number of purchase orders by invoice status.", sql="SELECT invoice_status, COUNT(*) FROM purchase_order GROUP BY invoice_status;")
        vn.train(question="Which purchase orders have notes?", sql="SELECT name FROM purchase_order WHERE notes IS NOT NULL AND notes <> '';")
        vn.train(question="Find the earliest effective date of any purchase order.", sql="SELECT MIN(effective_date) FROM purchase_order;")
        vn.train(question="List purchase orders where no mail flags are set to true.", sql="SELECT name FROM purchase_order WHERE mail_reminder_confirmed = false AND mail_reception_confirmed = false AND (mail_reception_declined = false OR mail_reception_declined IS NULL);")
        vn.train(question="Which purchase orders were confirmed by mail?", sql="SELECT name FROM purchase_order WHERE mail_reminder_confirmed = true;")
        vn.train(question="How many unique partners are associated with purchase orders?", sql="SELECT COUNT(DISTINCT partner_id) FROM purchase_order;")
        vn.train(question="List purchase orders grouped by user ID.", sql="SELECT user_id, COUNT(*) FROM purchase_order GROUP BY user_id;")
        vn.train(question="Which purchase orders have the name starting with P0000?", sql="SELECT name FROM purchase_order WHERE name LIKE 'P0000%';")
        vn.train(question="Which purchase orders have a null approval date?", sql="SELECT name FROM purchase_order WHERE date_approve IS NULL;")
        vn.train(question="How many purchase orders use incoterm ID 2?", sql="SELECT COUNT(*) FROM purchase_order WHERE incoterm_id = 2;")
        vn.train(question="List purchase orders ordered by amount total descending.", sql="SELECT name, amount_total FROM purchase_order ORDER BY amount_total DESC;")
        vn.train(question="What are the unique picking type IDs used in purchase orders?", sql="SELECT DISTINCT picking_type_id FROM purchase_order;")
        vn.train(question="Which purchase orders have a planned date on or after September 4, 2025?", sql="SELECT name FROM purchase_order WHERE date_planned >= '2025-09-04';")
        vn.train(question="Which purchase orders have the currency ID equal to 1?", sql="SELECT name FROM purchase_order WHERE currency_id = 1;")
        vn.train(question="What are the different receipt statuses used?", sql="SELECT DISTINCT receipt_status FROM purchase_order;")
        vn.train(question="List all purchase orders where the origin is not specified.", sql="SELECT name FROM purchase_order WHERE origin IS NULL OR origin = '';")
        vn.train(question="Which purchase orders were created by user 1 and are in draft state?", sql="SELECT name FROM purchase_order WHERE create_uid = 1 AND state = 'draft';")
        vn.train(question="How many purchase orders are associated with group ID 1?", sql="SELECT COUNT(*) FROM purchase_order WHERE group_id = 1;")
        vn.train(question="What is the total count of purchase orders with null group ID?", sql="SELECT COUNT(*) FROM purchase_order WHERE group_id IS NULL;")
        # Purchase order line
        vn.train(question="What is the total quantity ordered for product ID 5 in purchase orders?", sql="SELECT SUM(product_qty) FROM purchase_order_line WHERE product_id = 5;")
        vn.train(question="Which purchase order lines belong to order ID 1?", sql="SELECT id, name FROM purchase_order_line WHERE order_id = 1;")
        vn.train(question="What is the total price for each purchase order line in order ID 2?", sql="SELECT id, name, price_total FROM purchase_order_line WHERE order_id = 2;")
        vn.train(question="List all purchase order lines where the state is draft.", sql="SELECT id, name FROM purchase_order_line WHERE state = 'draft';")
        vn.train(question="What is the average unit price of products across all purchase order lines?", sql="SELECT AVG(price_unit) FROM purchase_order_line;")
        vn.train(question="Which purchase order lines have a discount applied?", sql="SELECT id, name, discount FROM purchase_order_line WHERE discount IS NOT NULL AND discount > 0;")
        vn.train(question="What is the total tax across all purchase order lines?", sql="SELECT SUM(price_tax) FROM purchase_order_line;")
        vn.train(question="How many units were received for product ID 6?", sql="SELECT SUM(qty_received) FROM purchase_order_line WHERE product_id = 6;")
        vn.train(question="Which purchase order lines are downpayments?", sql="SELECT id, name FROM purchase_order_line WHERE is_downpayment = true;")
        vn.train(question="List all unique product IDs used in purchase order lines.", sql="SELECT DISTINCT product_id FROM purchase_order_line WHERE product_id IS NOT NULL;")
        vn.train(question="Which purchase order lines have a planned date after September 3, 2025?", sql="SELECT id, name FROM purchase_order_line WHERE date_planned > '2025-09-03';")
        vn.train(question="What is the quantity to invoice for purchase order ID 1?", sql="SELECT SUM(qty_to_invoice) FROM purchase_order_line WHERE order_id = 1;")
        vn.train(question="Which purchase order lines have no tax?", sql="SELECT id, name FROM purchase_order_line WHERE price_tax = 0 OR price_tax IS NULL;")
        vn.train(question="What is the total price for product ID 32?", sql="SELECT SUM(price_total) FROM purchase_order_line WHERE product_id = 32;")
        vn.train(question="How many purchase order lines were created by user ID 1?", sql="SELECT COUNT(*) FROM purchase_order_line WHERE create_uid = 1;")
        vn.train(question="Which purchase order lines have a display type set?", sql="SELECT id, name, display_type FROM purchase_order_line WHERE display_type IS NOT NULL;")
        vn.train(question="List all purchase order lines with a null product packaging ID.", sql="SELECT id, name FROM purchase_order_line WHERE product_packaging_id IS NULL;")
        vn.train(question="Which purchase order lines have more than 10 units ordered?", sql="SELECT id, name, product_qty FROM purchase_order_line WHERE product_qty > 10;")
        vn.train(question="What is the subtotal price of each purchase order line in order ID 1?", sql="SELECT id, name, price_subtotal FROM purchase_order_line WHERE order_id = 1;")
        vn.train(question="Which purchase order lines were written after September 1, 2025?", sql="SELECT id, name FROM purchase_order_line WHERE write_date > '2025-09-01';")
        vn.train(question="What is the total quantity invoiced for product ID 33?", sql="SELECT SUM(qty_invoiced) FROM purchase_order_line WHERE product_id = 33;")
        vn.train(question="List all purchase order lines with a product UOM ID of 1.", sql="SELECT id, name FROM purchase_order_line WHERE product_uom = 1;")
        vn.train(question="Which purchase order lines are associated with partner ID 9?", sql="SELECT id, name FROM purchase_order_line WHERE partner_id = 9;")
        vn.train(question="How many purchase order lines use the stock_moves method?", sql="SELECT COUNT(*) FROM purchase_order_line WHERE qty_received_method = 'stock_moves';")
        vn.train(question="What is the total value of purchased products ordered (price_total)?", sql="SELECT SUM(price_total) FROM purchase_order_line;")
        vn.train(question="Which purchase order lines have a planned date before September 3, 2025?", sql="SELECT id, name FROM purchase_order_line WHERE date_planned < '2025-09-03';")
        vn.train(question="List the names of all products ordered in purchase order ID 2.", sql="SELECT name FROM purchase_order_line WHERE order_id = 2;")
        vn.train(question="What is the average discount given across all lines of purchase orders?", sql="SELECT AVG(discount) FROM purchase_order_line WHERE discount IS NOT NULL;")
        vn.train(question="Which purchase order lines have product packaging quantity set?", sql="SELECT id, name, product_packaging_qty FROM purchase_order_line WHERE product_packaging_qty IS NOT NULL AND product_packaging_qty > 0;")
        vn.train(question="List all purchase order lines with a non-null product description variant.", sql="SELECT id, name, product_description_variants FROM purchase_order_line WHERE product_description_variants IS NOT NULL;")
        vn.train(question="Which purchase order lines have propagate_cancel set to true?", sql="SELECT id, name FROM purchase_order_line WHERE propagate_cancel = true;")
        vn.train(question="How many products were ordered in total across all lines?", sql="SELECT SUM(product_qty) FROM purchase_order_line;")
        vn.train(question="Which purchase order lines are linked to sale order ID 1?", sql="SELECT id, name FROM purchase_order_line WHERE sale_order_id = 1;")
        vn.train(question="Which purchase order lines have a null sale order ID?", sql="SELECT id, name FROM purchase_order_line WHERE sale_order_id IS NULL;")
        vn.train(question="How many units have been manually received?", sql="SELECT SUM(qty_received_manual) FROM purchase_order_line;")
        vn.train(question="List all purchase order lines with product ID 33 and qty greater than 3.", sql="SELECT id, name FROM purchase_order_line WHERE product_id = 33 AND product_qty > 3;")
        vn.train(question="Which purchase order lines were created before September 2, 2025?", sql="SELECT id, name FROM purchase_order_line WHERE create_date < '2025-09-02';")
        vn.train(question="Get the planned date for each purchase order line in order ID 1.", sql="SELECT id, name, date_planned FROM purchase_order_line WHERE order_id = 1;")
        vn.train(question="What is the maximum unit price across all purchase order lines?", sql="SELECT MAX(price_unit) FROM purchase_order_line;")
        vn.train(question="Which purchase order lines have a subtotal over 1000?", sql="SELECT id, name, price_subtotal FROM purchase_order_line WHERE price_subtotal > 1000;")
        vn.train(question="List all purchase order lines with zero quantity ordered.", sql="SELECT id, name FROM purchase_order_line WHERE product_qty = 0;")
        # CRM stage
        vn.train(question="What are all the stages in the CRM pipeline?", sql="SELECT id, name FROM public.crm_stage ORDER BY sequence;")
        vn.train(question="Which stages are marked as 'Won'?", sql="SELECT id, name FROM public.crm_stage WHERE is_won = true;")
        vn.train(question="What is the sequence order of the CRM stages?", sql="SELECT id, name, sequence FROM public.crm_stage ORDER BY sequence;")
        vn.train(question="Which stages are folded by default?", sql="SELECT id, name FROM public.crm_stage WHERE fold = true;")
        vn.train(question="How many CRM stages are there?", sql="SELECT COUNT(*) FROM public.crm_stage;")
        vn.train(question="Which users created each stage?", sql="SELECT s.id, s.name, u.login AS created_by FROM public.crm_stage s JOIN public.res_users u ON s.create_uid = u.id;")
        vn.train(question="Which stages have specific requirements filled in?", sql="SELECT id, name, requirements FROM public.crm_stage WHERE requirements IS NOT NULL AND requirements <> '';")
        vn.train(question="What is the most recently created stage?", sql="SELECT id, name, create_date FROM public.crm_stage ORDER BY create_date DESC LIMIT 1;")
        vn.train(question="Which stages were updated after creation?", sql="SELECT id, name, create_date, write_date FROM public.crm_stage WHERE write_date > create_date;")
        vn.train(question="Show all stages along with their team ID.", sql="SELECT id, name, team_id FROM public.crm_stage;")

        print("Vanna training completed!")
        # Test the connection and data visibility
        try:
            test_sql = vn.generate_sql("List all categories")
            print(f"Test SQL generated: {test_sql}")
        except Exception as e:
            print(f"Test query failed: {e}")

    except Exception as e:
        print(f"Training error: {e}")
else:
    print("Warning: OPENAI_API_KEY not set, running in demo mode")

# --- Create Flask app ---
app = Flask(__name__, static_folder='static')
print('created flask app line 168')
# Configure CORS with more permissive settings
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True)
print('=>=>=> executed cors')
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

    chat_logger = logging.getLogger('chat_logger')
    chat_file_handler = logging.FileHandler("./debug"+'/'+file_name+".log")  # Log file name
    chat_file_handler.setLevel(logging.DEBUG)
    chat_logger.addHandler(chat_file_handler)

    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response
    
    try:
        try:
            data = request.get_json(force=True)  # Force JSON parsing
            chat_logger.info("data")
            chat_logger.info(json.dumps(data,indent=3))
            print(data.get("mode"))
            print("="*500)
        except Exception as e:
            chat_logger.error(f"JSON parsing error: {e}")
            return jsonify({"error": {"message": "Invalid JSON format", "type": "invalid_request"}}), 400

        if not data:
            chat_logger.info("Inside not data")
            return jsonify({"error": {"message": "No JSON data received", "type": "invalid_request"}}), 400

        messages = data.get('messages', [])
        # logger.info("messages")
        # for message in messages:
        #    logger.info(json.dumps(message,indent=3))
        model = data.get('model', 'vanna-sql-query')
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
            chat_logger.info("not user_messages")
            return jsonify({"error": {"message": "No user message found", "type": "invalid_request"}}), 400

        question = user_messages[-1].get('content', '').strip()
        # logger.info("question")
        # here is the models question
        # logger.info(">"*10+question+"<"*10)
        # Generate response
        if vn and question and not question.lower().startswith('hello'):
            try:
                chat_logger.info("vn and question and not question.lower().startswith('hello')")
                # Ensure allow_llm_to_see_data is True
                vn.allow_llm_to_see_data = True

                # Try to generate SQL and get results
                sql = vn.generate_sql(question, allow_llm_to_see_data=True)
                chat_logger.info("sql anch")
                # here is the generate sql by the model
                chat_logger.info(sql)
                if isinstance(sql, str) and sql.strip().upper().startswith("SELECT"):
                    # here is the models results.
                    result = vn.run_sql(sql)
                    chat_logger.debug("result of run_sql")
                    # logger.debug(f"The type of the returned results of the run_sql is of type {type(result)}")
                    chat_logger.info(result)
                    # response_text = f"Here's the SQL query I generated for your question:\n\n```sql\n{sql}\n```\n\nResults:\n{result.to_string() if hasattr(result, 'to_string') else str(result)}"
                    response_text = f"Here's the SQL query I generated for your question:\n\n```sql\n{sql}\n```\n\nResults:\n{result.to_markdown(index=False)}"

                else:
                    response_text = f"I generated this response: {sql}"
                    chat_logger.info("response_text")
                    chat_logger.info(response_text)
            except Exception as e:
                response_text = f"I encountered an error while processing your question: {str(e)}\n\nTip: Make sure your question is related to the data the model has knowledge about."
                chat_logger.error("response_text")
                chat_logger.error(response_text)
        else:
            # Default responses for testing or when Vanna is not available
            if question.lower().startswith('hello') or not question:
                response_text = "Hello! I'm your SQL assistant. I can help you query sales data. Try asking questions like:\n- 'List all products'\n- 'Show me Alice's purchases'\n- 'Which customer spent the most?'\n- 'What's the total revenue?'"
            else:
                response_text = f"I received your question: '{question}'. This is a test response showing that the connection is working."
            chat_logger.error("response_text")
            chat_logger.error(response_text)
        # Handle streaming vs non-streaming
        if stream:
            chat_logger.debug("Inside stream")
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
                chat_logger.debug("chunk_data")
                chat_logger.debug(json.dumps(chunk_data,indent=3))
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
                chat_logger.info("final_chunk")
                chat_logger.info(json.dumps(final_chunk))
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
        chat_logger.error(f"Error in chat_completions: {e}")
        return jsonify({
            "error": {
                "message": f"Internal server error: {str(e)}",
                "type": "internal_error"
            }
        }), 500


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
            "id": "vanna-sql-query",
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
    print(f"3. Select model 'vanna-sql-query' in the chat interface")
    print(f"4. Test with: 'Hello' or 'List all products'")
    print(f"\n=== URLs ===")
    print(f"- API: http://localhost:{API_PORT}")
    print(f"- Test endpoint: http://localhost:{API_PORT}/test")
    print(f"- Models: http://localhost:{API_PORT}/v1/models")
    print(f"==========================")

    # Run Flask app
    # app.run(host="0.0.0.0", port=API_PORT, debug=True)
