### SqlDatabase Chain by Langchain
In this repository I have used SqlDatabaseChain of Langchain. Data is present in the excel file. Requirement you have to load the data into a sql database. 
Connect to this Sql Database then we proceed. 
I have used FewShot Prompting to enhance the SqlQuery generated by # SqlDatabaseChain. 
Examples are present in the feedback.yaml file.
I have build it using openai and starchatbeta(Loading using HuggingFace). 
The main code is present in the mainv5.py file. This code uses the capability of SqlDatabaseChain from Langchain to generate the sql query(with the help of examples provided) step by step and then fire the query on the database to retrieve the answer.
