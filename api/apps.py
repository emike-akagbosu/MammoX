from django.apps import AppConfig


class ApiConfig(AppConfig):
    '''Defines secondary app'''
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'


