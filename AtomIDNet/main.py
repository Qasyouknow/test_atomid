from torchvision import transforms
import time
import torch
from utils.data_utils import *
from PIL import Image
from tqdm import tqdm
from models.unet import get_model

DEMO_PATH = r".\datasets\testset"

def get_demo_Data(path:str=DEMO_PATH, endwith:str='.bmp'):
    names = getFileList(path, endwith)
    images = getImgList(path, endwith)
    
    images = [Image.fromarray(img) for img in images]
    names = [name.split('\\')[-1].replace(endwith, '') for name in names]
    return zip(images, names)


if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model().to(device)
    
    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])
    
    demo_loader = get_demo_Data()
    
    results_save_dir = r'D:\PythonProjects\pythonProject\PolyU\FYP\AtomIDNet\result_test'
    # Demo
    model.eval()
    epoch_loss = []
    test_start = time.time()
    for image, name in tqdm(demo_loader):
        inputs = normalize(toTensor(image)).unsqueeze(0)
        inputs = inputs.type(torch.FloatTensor).to(device)
        _, _, h, w = inputs.size()
        inputs = F.interpolate(inputs, size=(round(h/16) * 16, round(w/16) * 16))
        
        out_diam, out_map = model(inputs)
        keep_centers = visualize(inputs, out_diam, out_map, None, [name], save_path=results_save_dir)


        keep_centers = keep_centers.squeeze().cpu().numpy().astype('int')

        csv_name = rf'{results_save_dir}\{name}.csv'
        with open(csv_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Peak #', 'X', 'Y'])
            for idx, c in enumerate(keep_centers):
                csv_writer.writerow([str(idx)] + [str(int(round(c[0])))] + [str(int(round(c[1])))])