import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cvlib
import matplotlib.pyplot as plt
import sys


def img_pad(img, r):
    """Recibe una imagen en blaco y negro y le agrega un padding de ceros pixeles a su alrededor 

    input: 
        - img (np.array): imagen en blanco y negro 
        - r (int): ancho del padding

    return: 
        - img_result(np.array): imagen con padding de ceros 
    
    """
    # inicio de función
    f, c = img.shape

    f, c, r

    img_result = np.zeros((f+2*r, c+2*r), dtype=int)
    img_result


    for i in range(r, f + r):
        for j in range(r, c + r): 
            img_result[i, j] = img[i-r, j-r]

    return img_result 
    # fin de función



class UnionFind:
    """ Soluciona las relaciones entre los grupos

    Implementación tomada de: https://yuminlee2.medium.com/union-find-algorithm-ffa9cd7d2dba#6382  
    """
    def __init__(self, numOfElements):
        """ 
        input: 
             - numOfElements(int) = numero de elementos
        """
        self.parent = self.makeSet(numOfElements)
        self.size = [1]*numOfElements
        self.count = numOfElements
    
    def makeSet(self, numOfElements):
        return [x for x in range(numOfElements)]

    # Time: O(logn) | Space: O(1)
    def find(self, node):
        """ Encuentra el valor por el que nodo debe ser separado

        Args:
            node (int): nodo de entrada

        Returns:
            int: el nodo por el que debe ser mapeado 
        """
        while node != self.parent[node]:
            # path compression
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    # Time: O(1) | Space: O(1)
    def union(self, node1, node2):
        """ Recibe las relaciones

        Args:
            node1 (int): nodo 1 de la relación
            node2 (int): nodo 1 de la relación
        """
        root1 = self.find(node1)
        root2 = self.find(node2)

        # already in the same set
        if root1 == root2:
            return

        if self.size[root1] > self.size[root2]:
            self.parent[root2] = root1
            self.size[root1] += 1
        else:
            self.parent[root1] = root2
            self.size[root2] += 1
        
        self.count -= 1



def connected_c(img):
    """ genera grupos de pixeles por medio del algoritmo 4-connected y da como resultado una matriz con los grupos 

    input: 
        - img (matriz de int): imagen binarizada para encontrar las relaciones
    
    output: 
        - res (matriz de int): matriz con las agrupaciones de pixeles. 
    """
    # INICIO de funcion 


    # inicio primer recorrido 
    f, c = img.shape
    res = np.zeros((f, c), dtype=int)
    index = 0
    relaciones = {}
    relaciones_lista = []

    for i in range(1, f - 1):
        # print("Fila:", i)
        for j in range(1, c-1): 
            # print("\tColumna:", j)

            centro = img[i, j]
            upper = res[i - 1, j] # valor de arriba
            left = res[i, j-1] # valor de la izquierda

            if centro == 255:
                # valor por defecto
                value = 0

                if (upper !=0 and left !=0):
                    # obtencion del indice más pequeño 
                    value = min(upper, left)
                    
                    # creacion de relacion el mayor es la key y el más pequeño es el value
                    if upper != left:
                        relaciones[max(upper, left)] = value
                        relaciones_lista.append([upper, left])

                elif (upper != 0):
                    value = upper 
                
                elif (left != 0):
                    value = left

                else:
                    index += 1
                    value = index
                
                res[i, j] = value 
    # fin primer recorrido



    # incio resolución de relaciones 
    uf = UnionFind(index + 1)

    for node1, node2 in relaciones_lista:
        uf.union(node1, node2)

    nuevas_relaciones = {}

    for i in range(index + 1): 
        nuevas_relaciones[i] = uf.find(i)
        
    # nuevas_relaciones  = find_relaciones2(index, relaciones_lista)
    # fin resolución de relaciones 


    # inicio segundo recorrido

    for i in range(0, f):
        for j in range(0, c):
            actual_value = res[i, j]
            res[i, j] = nuevas_relaciones[actual_value]
    # fin segundo recorrido

    return res
    # FIN de funcion



def get_random_color():
    """ Genera un color aleatorio en el formato RGB 
    input:
        - None 
    output:
        -  (arrat de int): array con un color en formato RGB [-,-,-]
    """

    return list(np.random.choice(range(256), size=3))



def labelview(img, seed = None, filename = None, size = 10):
    """ Recibe una matriz con agrupaciones de pixeles le asigna un color en formato RGB a cada grupo y prientea la imagen

    input: 
        - img (matriz de int): matriz con las agrupaciones de pixeles por medio de numeros.
        - seed (int): semilla para la asignación de numeros aleatorios. (por defecto None)
        - filename (string): nombre del archivo para que e guarde la imagen 
    """

    if (seed != None): 
        np.random.seed(seed)

    components = np.unique(img)
    colors = {0:[0, 0, 0]}
    # print(components)

    # creacion de colores random
    for i in components:
        if i != 0: 
            colors[i] = get_random_color()
            # print(i, colors[i])        
    
    # pprint(colors)

    # colorear la imagen 
    f, c = img.shape

    img_res = []

    for i in range(0, f):
        new_row = []

        for j in range(0, c): 
            # se asigna el color que corresponde a su segmento
            segment = img[i, j]
            new_row.append(colors[segment])

        img_res.append(new_row)

    img_res = np.array(img_res)

    if filename is not None: 
        cvlib.imgview(img_res, size=size, filename = filename)
    else:
        cvlib.imgview(img_res, size=size)

    # return img_res



def main_function(input_filename = "fprint3.pgm", output_filename = "output.png"):
    """Funcion main para ejecutar el laboratorio

    Args:
        input_filename (str, optional): nombre de la imagen de entrada. Defaults to "fprint3.pgm".
        output_filename (str, optional): nombre de la imagen de salida. Defaults to "output.png".
    """

    # lectura de la imagen en blanco y negro 
    img = cv.imread(input_filename, cv.IMREAD_GRAYSCALE)

    # Blur para suaviar las lineas de la huella 
    b = 5
    img_blur = cv.blur(img,(b,b))


    # Aplicación de flood fill para eliminar el ruido de la imagen 
    img3 = img_blur.copy()
    D = 22
    img3 = cv.floodFill(img3, None, (40, 40), 255,  loDiff=D, upDiff=D, flags=cv.FLOODFILL_FIXED_RANGE)[1]


    # Binzarización inversa con Adaptative Treshold par mejorar  
    imgbin = cv.adaptiveThreshold(img3, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,31,5)


    img_padded = img_pad(imgbin, 1)  ## PADDING 
    resultado = connected_c(img_padded)  ## CONCECTED_COMPONENTS
    labelview(resultado, size=100, filename = output_filename) ## LABEL_VIEW 


args = sys.argv
input_filename =  "fprint3.pgm"
output_filename = "output.png"

try:
    input_filename = args[1]
except:
    pass

try:
    output_filename = args[2]
except:
    pass

print("Procesando imagen:", input_filename, "...")

main_function(input_filename, output_filename)

print("Procesando finalizado")
print("Output filename:", output_filename)
