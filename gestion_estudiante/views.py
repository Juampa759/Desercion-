import io
from os import curdir, sep
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from gestion_estudiante.forms import formulario_usuario
from gestion_estudiante.models import Usuario
from django.urls import reverse, reverse_lazy
from django.contrib import messages 
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.contrib.messages.views import SuccessMessageMixin 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

def redes (request):
    
    dataset = pd.read_csv('Bancos.csv')
    x = dataset.iloc[:, 3:13].values 
    y = dataset.iloc[:,13].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_x_1 = LabelEncoder()
    x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
    labelencoder_x_2 = LabelEncoder()
    x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

    from sklearn.compose import ColumnTransformer
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = np.array(columnTransformer.fit_transform(x), dtype = np.float)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    modelo = './modelo/modelo.h5'
    pesos_modelo = './modelo/pesos.h5'
    clasificador = load_model(modelo)
    clasificador.load_weights(pesos_modelo)

    y_pred = clasificador.predict(x_test)
    y_pred = (y_pred>0.5)

    # Matriz de Confusion
    from sklearn.metrics import confusion_matrix
    mc = confusion_matrix(y_test,y_pred)

    #probando la red neuronal
    nuevo_cliente = clasificador.predict(sc.transform(np.array([[0,1,600,0,40,3,60000,2,1,1,50000]])))


    return HttpResponse(nuevo_cliente)


# Create your views here.

"""def usuario(request):

    if request.method=="POST":

        miform=formulario_usuario(request.POST)

        if miform.is_valid():
            add = Usuario(geografia = 'hola')
            add.save()
            #return render(request, "estadistica.html")
    else:

        miform=formulario_usuario()

    return render(request, "formulario_usuario.html",{"form":miform})"""

def create(request):
    if request.method == 'POST':
        form = formulario_usuario(request.POST)
        if form.is_valid():
            form.save()

            dataset = pd.read_csv('Bancos.csv')
            x = dataset.iloc[:, 3:13].values 
            y = dataset.iloc[:,13].values

            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            labelencoder_x_1 = LabelEncoder()
            x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
            labelencoder_x_2 = LabelEncoder()
            x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

            from sklearn.compose import ColumnTransformer
            columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2])], remainder='passthrough')
            x = np.array(columnTransformer.fit_transform(x), dtype = np.float)

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            modelo = './modelo/modelo.h5'
            pesos_modelo = './modelo/pesos.h5'
            clasificador = load_model(modelo)
            clasificador.load_weights(pesos_modelo)

            y_pred = clasificador.predict(x_test)
            y_pred = (y_pred>0.5)

            # Matriz de Confusion
            from sklearn.metrics import confusion_matrix
            mc = confusion_matrix(y_test,y_pred)

            #probando la red neuronal
            """
            Geografía: Francia
            Puntaje de crédito: 600
            Género: Masculino
            Edad: 40 años
            Tenencia: 3 años
            Saldo: $60000
            Número de productos: 2
            ¿Tiene este cliente una tarjeta de crédito? Sí
            ¿Este cliente es un miembro activo? Si
            Salario estimado: $50000
            """
            geo = request.POST['geografia']
            puncre = request.POST['puntajeCre']
            gene = request.POST['genero']
            edad = request.POST['edad']
            tene = request.POST['tenencia']
            Sald = request.POST['saldo']
            numpro = request.POST['numproduc']
            tarCre = request.POST['tarCredito']
            acti = request.POST['activo']
            salar = request.POST['salario']

            nuevo_cliente = clasificador.predict(sc.transform(np.array([[0,1,puncre,gene,edad,tene,Sald,numpro,tarCre,acti,salar]])))


            return HttpResponse( nuevo_cliente )
    else:
        form = formulario_usuario()

    return render(request, 'formulario_usuario.html',{'form': form})

class IndexView(ListView):
    template_name = 'index.html'
    context_object_name = 'usuario_list'

    def get_queryset(self):
        return Usuario.objects.all()

class Actualizar(SuccessMessageMixin, UpdateView): 
    model = Usuario
    form = formulario_usuario 
    fields = "__all__"  
    success_message = 'Usuario Actualizado Correctamente !' 
 

    def get_success_url(self):               
        return redirect('/index/') 

class Eliminar(DeleteView):
    model = Usuario
    success_url = reverse_lazy('usuario_list')

def inicio(request):
    return render(request, 'inicio.html')

def acerca(request):
    return render (request, 'Acerca.html')

"""
def Eliminar(request, pk, template_name='crudapp/confirm_delete.html'):
    contact = get_object_or_404(Usuario, pk=pk)
    if request.method=='POST':
        contact.delete()
        return redirect('index')
    return render(request, template_name, {'object':contact})"""

