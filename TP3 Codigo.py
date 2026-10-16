import random

class RedHopfieldIndustrial:
    def __init__(self, filas=10, columnas=10):
        self.filas = filas
        self.columnas = columnas
        self.n = filas * columnas  # 100 neuronas
        self.W = [[0.0 for _ in range(self.n)] for _ in range(self.n)]

    def entrenar_hebb(self, patrones):
        self.W = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        vectores = [[val for fila in p for val in fila] for p in patrones]
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.W[i][j] = sum(v[i] * v[j] for v in vectores)
                else:
                    self.W[i][j] = 0.0

    def entrenar_pseudoinversa(self, patrones):
        self.W = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        q = len(patrones)
        X = [[p[r][c] for p in patrones] for r in range(self.filas) for c in range(self.columnas)]
        # M = X^T * X (matriz de q x q)
        M = [[0.0 for _ in range(q)] for _ in range(q)]
        for a in range(q):
            for b in range(q):
                M[a][b] = sum(X[i][a] * X[i][b] for i in range(self.n))
        
        if q == 1:
            inv_M = [[1.0 / M[0][0]]]
        elif q == 2:
            det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
            if abs(det) < 1e-9:
                raise ValueError("Patrones linealmente dependientes.")
            inv_det = 1.0 / det
            inv_M = [
                [ M[1][1] * inv_det, -M[0][1] * inv_det],
                [-M[1][0] * inv_det,  M[0][0] * inv_det]
            ]
        else:
            raise NotImplementedError("Soporte analitico para q <= 2.")
        # W = X * inv_M * X^T 
        for i in range(self.n):
            for j in range(self.n):
                val = 0.0
                for a in range(q):
                    for b in range(q):
                        val += X[i][a] * inv_M[a][b] * X[j][b]
                self.W[i][j] = val 

    def calcular_energia(self, estado):
        v = [val for fila in estado for val in fila]
        suma = 0.0
        for i in range(self.n):
            for j in range(self.n):
                suma += self.W[i][j] * v[i] * v[j]
        return -0.5 * suma

    def recuperar(self, patron_ruidoso, max_ciclos=10):
        estado = [val for fila in patron_ruidoso for val in fila]
        def a_matriz(vec):
            return [vec[r * self.columnas : (r + 1) * self.columnas] for r in range(self.filas)]
        historial_energia = [self.calcular_energia(a_matriz(estado))]
        for _ in range(max_ciclos):
            for i in range(self.n):
                potencial = sum(self.W[i][j] * estado[j] for j in range(self.n))
                estado[i] = 1 if potencial >= 0 else -1
            mat_act = a_matriz(estado)
            e_act = self.calcular_energia(mat_act)
            historial_energia.append(e_act)

            if abs(historial_energia[-1] - historial_energia[-2]) < 1e-4:
                break

        return a_matriz(estado), historial_energia
    
def generar_patron_anillo_y_escuadra():
    img = [[-1 for _ in range(10)] for _ in range(10)]
    # Aro 'C' (rombo regular centrado en filas 1 a 6)
    anillo_coords = [
        (1, 4), (1, 5),
        (2, 3), (2, 6),
        (3, 2), (3, 7),
        (4, 2), (4, 7),
        (5, 3), (5, 6),
        (6, 4), (6, 5)
    ]
    for r, c in anillo_coords:
        img[r][c] = 1
    # Escuadra fija (esquina inferior izquierda)
    escuadra_coords = [(8, 0), (9, 0), (9, 1)]
    for r, c in escuadra_coords:
        img[r][c] = 1
    return img

def aplicar_ruido_y_defectos(patron, prob_ruido=0.20, prob_defecto=0.25):
    random.seed(42)
    corrupto = [fila[:] for fila in patron]
    for r in range(10):
        for c in range(10):
            if (r, c) in [(8, 0), (9, 0), (9, 1)]:
                continue
            if patron[r][c] == 1:
                if random.random() < prob_defecto:
                    corrupto[r][c] = -1
            else:
                if random.random() < prob_ruido:
                    corrupto[r][c] = 1
    return corrupto

def calcular_centro_aro(img):
    coords_aro = []
    for r in range(10):
        for c in range(10):
            if img[r][c] == 1 and not (r >= 8 and c <= 1):
                coords_aro.append((r, c))
    if not coords_aro:
        return None, None
    prom_y = sum(r for r, c in coords_aro) / float(len(coords_aro))
    prom_x = sum(c for r, c in coords_aro) / float(len(coords_aro))
    return prom_x, prom_y

def imprimir_matriz(img, titulo):
    print(f"\n--- {titulo} ---")
    for fila in img:
        print(" " + " ".join(["X" if val == 1 else "." for val in fila]))
if __name__ == "__main__":
    print("=" * 70)
    print("PROTOTIPO HOPFIELD - PLANTA DE MOTORES")
    print("=" * 70)

    patron_ideal = generar_patron_anillo_y_escuadra()
    patrones_memoria = [patron_ideal]
    patron_ruidoso = aplicar_ruido_y_defectos(patron_ideal)

    imprimir_matriz(patron_ideal, "Patron Ideal de Referencia (Aro C + Escuadra)")
    imprimir_matriz(patron_ruidoso, "Imagen Degradada de Entrada (Ruido + Defectos)")

    # 1. Regla de Hebb
    red_hebb = RedHopfieldIndustrial()
    red_hebb.entrenar_hebb(patrones_memoria)
    salida_hebb, energia_hebb = red_hebb.recuperar(patron_ruidoso)
    imprimir_matriz(salida_hebb, "Salida Recuperada: Regla de Hebb")

    # 2. Matriz Pseudoinversa
    red_pseudo = RedHopfieldIndustrial()
    red_pseudo.entrenar_pseudoinversa(patrones_memoria)
    salida_pseudo, energia_pseudo = red_pseudo.recuperar(patron_ruidoso)
    imprimir_matriz(salida_pseudo, "Salida Recuperada: Matriz Pseudoinversa")

    # 3. Cálculo de Coordenadas
    cx_ideal, cy_ideal = calcular_centro_aro(patron_ideal)
    cx_pseudo, cy_pseudo = calcular_centro_aro(salida_pseudo)

    print("\n" + "=" * 70)
    print("METRICAS COMPARATIVAS DE DESEMPEÑO")
    print("=" * 70)
    print(f"Energia inicial en entrada ruidosa: {energia_pseudo[0]:.2f}")
    print(f"Ciclos de relajacion (Hebb):        {len(energia_hebb)-1} ciclos")
    print(f"Energia final alcanzada (Hebb):     {energia_hebb[-1]:.2f}")
    print(f"Ciclos de relajacion (Pseudoinv):   {len(energia_pseudo)-1} ciclos")
    print(f"Energia final alcanzada (Pseudoinv):{energia_pseudo[-1]:.2f}")
    print("-" * 70)
    print(f"Centro teorico ideal Aro 'A':       X = {cx_ideal:.2f}, Y = {cy_ideal:.2f}")
    print(f"Centro inferido tras Pseudoinversa: X = {cx_pseudo:.2f}, Y = {cy_pseudo:.2f}")
    print(f"Precision metrica de localizacion:  {'100% Exacta' if (cx_ideal==cx_pseudo and cy_ideal==cy_pseudo) else 'Aproximada'}")
    print("=" * 70)