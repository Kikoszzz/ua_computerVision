import numpy as np
import open3d as o3d
import cv2
import os
import sys

def main():
    # 1. Caminhos dos arquivos (ajuste se necessário)
    points_file = 'points3d.npz'
    # Usaremos a imagem original para dar cor aos pontos
    # Se você tiver a imagem retificada, o resultado será muito melhor
    image_file = '../images/left01.jpg' 

    if not os.path.exists(points_file):
        print(f"Erro: Arquivo {points_file} não encontrado.")
        sys.exit(1)

    # 2. Carregar dados
    data = np.load(points_file)
    points_3D = data['points3d']
    
    # Carregar imagem para textura
    img = cv2.imread(image_file)
    if img is None:
        print(f"Aviso: Imagem {image_file} não encontrada. Criando nuvem sem cor.")
        colors_3D = None
    else:
        # Converter BGR para RGB e normalizar [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colors_3D = img_rgb.astype(np.float32) / 255.0

    # 3. Preparar os arrays (Flattening)
    p = points_3D.reshape(-1, 3)
    
    # 4. Filtragem de NaNs e Infs
    # Criamos uma máscara booleana para filtrar pontos inválidos de uma vez
    mask = np.isfinite(p).all(axis=1)
    fp = p[mask]
    
    if colors_3D is not None:
        fc = colors_3D.reshape(-1, 3)[mask]
    
    # Verificar se restaram pontos após a filtragem
    if len(fp) == 0:
        print("Erro: Nenhum ponto válido (finite) encontrado nos dados 3D.")
        sys.exit(1)

    # 5. Criar PointCloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(fp)
    
    if colors_3D is not None:
        pcl.colors = o3d.utility.Vector3dVector(fc)

    # 6. Cropping (Filtragem no eixo Z entre 0.1 e 2.0 ou 5.0)
    # Definimos os limites manualmente para evitar erros com bounding boxes vazias
    z_min, z_max = 0.1, 2.0
    
    # Criar uma bounding box alinhada aos eixos
    # O bounding box padrão pega o min/max dos dados, nós sobrescrevemos o Z
    bbox = pcl.get_axis_aligned_bounding_box()
    min_bound = np.copy(bbox.min_bound)
    max_bound = np.copy(bbox.max_bound)
    
    min_bound[2] = z_min
    max_bound[2] = z_max
    
    # Criar nova box com limites corrigidos
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcl_cropped = pcl.crop(crop_box)

    # 7. Visualização
    print(f"Pontos originais: {len(fp)}")
    print(f"Pontos após crop: {len(pcl_cropped.points)}")

    if len(pcl_cropped.points) == 0:
        print("Aviso: O filtro Z removeu todos os pontos. Mostrando nuvem original.")
        to_show = pcl
    else:
        to_show = pcl_cropped

    # Adicionar eixos coordenados
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    o3d.visualization.draw_geometries([to_show, axes],
                                      window_name="Visualização 3D - UA Computer Vision",
                                      width=1024,
                                      height=768)

if __name__ == "__main__":
    main()