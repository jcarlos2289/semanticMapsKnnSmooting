#Script para enviar los .tex del documento hacia el  repositorio de bitbucket
#By JCarlos

if test $# -lt 1 
	then 
	echo "Error: Introduce el Nombre para el Commit"
else
	git add .
	git commit -m $1
	git push -u origin master
	echo "Iniciando respaldo del commit " $1
	echo "Respaldo Terminado " 
fi
