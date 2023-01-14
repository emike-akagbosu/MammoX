# MammoX

App Instructions:
1. Create Virtual Environment on local device
Windows: 
1. Clone repo on local device in virtual environment 
2. Navigate into mammox directory
3. Run these commands:
    - pip install -r requirements.txt
    - python .\manage.py makemigrations
    - python .\manage.py migrate
    - python .\manage.py runserver

Note: If you clone the repository using Pycharm,
click 'OK' if given the option to automatically create
a virtual environment from the requirements.txt file.

macOS: 
1. Clone repo on local device
2. Navigate into mammoxdev-env/mammox directory
3. Run these commands:
    - python \manage.py makemigrations
    - python \manage.py migrate
    - python \manage.py runserver
