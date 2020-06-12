from django import forms
from gestion_estudiante.models import  Usuario

class formulario_usuario(forms.ModelForm):
    class Meta:
        model = Usuario
        fields= [
            'id',
            'geografia',
            'puntajeCre',
            'genero',
            'edad',
            'tenencia',
            'saldo',
            'numproduc',
            'tarCredito',
            'activo',
            'salario',
        ]
        labels = {
            'id': 'id',
            'geografia': 'Geografia',
            'puntajeCre': 'puntaje Credito',
            'genero': 'Genero',
            'edad': 'Edad',
            'tenencia': 'Tenencia',
            'saldo': 'Saldo',
            'numproduc': 'Numero del producto',
            'tarCredito': 'Tarjeta de credito',
            'activo': 'Activo',
            'salario': 'Salario',
        }
        widgets = {
            'id':forms.TextInput(attrs={'class':'form-control'}),
            'geografia':forms.TextInput(attrs={'class':'form-control'}),
            'puntajeCre':forms.TextInput(attrs={'class':'form-control'}),
            'genero':forms.TextInput(attrs={'class':'form-control'}),
            'edad':forms.TextInput(attrs={'class':'form-control'}),
            'tenencia':forms.TextInput(attrs={'class':'form-control'}),
            'saldo':forms.TextInput(attrs={'class':'form-control'}),
            'numproduc':forms.TextInput(attrs={'class':'form-control'}),
            'tarCredito':forms.TextInput(attrs={'class':'form-control'}),
            'activo':forms.TextInput(attrs={'class':'form-control'}),
            'salario':forms.TextInput(attrs={'class':'form-control'}),
        }