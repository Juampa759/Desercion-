# Generated by Django 3.0.6 on 2020-05-21 00:51

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Usuario',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('geografia', models.CharField(max_length=50)),
                ('puntajeCre', models.IntegerField()),
                ('genero', models.BooleanField()),
                ('edad', models.IntegerField()),
                ('tenencia', models.IntegerField()),
                ('saldo', models.IntegerField()),
                ('numproduc', models.IntegerField()),
                ('tarCredito', models.BooleanField()),
                ('activo', models.BooleanField()),
                ('salario', models.IntegerField()),
            ],
        ),
    ]
