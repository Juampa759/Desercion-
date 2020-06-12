from django.db import models

# Create your models here.
class Usuario(models.Model):
    id=models.IntegerField(primary_key=True)
    geografia=models.CharField(max_length=50)
    puntajeCre=models.IntegerField()
    genero=models.IntegerField()
    edad= models.IntegerField()
    tenencia= models.IntegerField()
    saldo= models.IntegerField()
    numproduc=models.IntegerField()
    tarCredito=models.IntegerField()
    activo=models.IntegerField()
    salario=models.IntegerField()

    def __str__(self):
        return self.geografia
