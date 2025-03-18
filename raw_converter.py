
from abc import abstractmethod

from PIL import Image
import numpy as np
import torch
import torch.nn as nn

import basicblock as B

try:
    import cv2
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
except:
    pass


class RawConverter():
    def __init__(self, bayer_type: str='grbg') -> None:
        self.bayer_type = bayer_type

    def convert(self, raw_img: np.ndarray, bayer_type: str=None) -> np.ndarray:
        """
        Args:
            raw_img: shape (H, W, 1) in range [0, 1]
        Return:
            rgb_img: shape (H, W, 3) in range [0, 1]
        """
        bayer_type = self.bayer_type if bayer_type is None else bayer_type
        return self._convert(raw_img, bayer_type)

    @abstractmethod
    def _convert(self, raw_img: np.ndarray, bayer_type: str) -> np.ndarray:
        raise NotImplementedError

    def read_raw(self, raw_file_path: str, black_level: int=0, raw_bit: int=10) -> np.ndarray:
        """
        Assume the raw image have shape (3060, 4080) or (6120, 8160)
        """
        # read raw image (shape: [h*w, 1])
        raw_img = np.fromfile(raw_file_path, dtype=np.uint16)

        # reshape
        if raw_img.shape[0] == 3060 * 4080:
            raw_img = raw_img.reshape(3060, 4080)
        elif raw_img.shape[0] == 6120 * 8160:
            raw_img = raw_img.reshape(6120, 8160)

        # int to float (range [0, 1])
        raw_img = raw_img.astype(np.float32)
        raw_img -= black_level
        raw_img /= (2**raw_bit - black_level)
        raw_img = np.clip(raw_img, 0, 1)

        return raw_img[..., None]


class BayerRawConverter(RawConverter):
    def __init__(self, bayer_type: str='grbg', dm_type: str='rstcanet', ckpt_path: str=None, device: str='cuda') -> None:
        super().__init__(bayer_type)

        assert dm_type in ['opencv', 'menon', 'rstcanet']
        self.dm_type = dm_type
        if dm_type == 'rstcanet':
            self.rstcanet = RSTCANet()
            assert ckpt_path is not None, f'ckpt_path should be provided for dm_type: {dm_type}'
            self.rstcanet.load_state_dict(torch.load(ckpt_path), strict=True)
            self.rstcanet.to(device)
            self.rstcanet.eval()
            self.device = device

    def _convert(self, raw_img, bayer_type):
        if self.dm_type == 'opencv': return self._convert_opencv(raw_img, bayer_type)
        elif self.dm_type == 'menon': return self._convert_menon(raw_img, bayer_type)
        elif self.dm_type == 'rstcanet': return self._convert_rstcanet(raw_img, bayer_type)

    def _convert_opencv(self, raw_img, bayer_type):
        if bayer_type.lower() == 'grbg': bayer_code = cv2.COLOR_BAYER_GR2RGB
        elif bayer_type.lower() == 'rggb': bayer_code = cv2.COLOR_BAYER_RG2RGB
        elif bayer_type.lower() == 'gbrg': bayer_code = cv2.COLOR_BAYER_GB2RGB
        elif bayer_type.lower() == 'bggr': bayer_code = cv2.COLOR_BAYER_BG2RGB
        else: raise NotImplementedError(f'Unknown bayer type: {bayer_type}')

        # raw_img: (H, W, 1)
        dm_img = (raw_img * 16383).astype(np.uint16)
        dm_img = cv2.cvtColor(dm_img, bayer_code)
        dm_img = dm_img.astype(np.float32) / 16383
        dm_img = dm_img.clip(0, 1)
        return dm_img

    def _convert_menon(self, raw_img, bayer_type):
        dm_img = demosaicing_CFA_Bayer_Menon2007(raw_img.squeeze(), pattern=bayer_type.upper())
        dm_img = dm_img.clip(0, 1)
        return dm_img

    @torch.no_grad()
    def _convert_rstcanet(self, raw_img, bayer_type):
        raw_img, h_idx, w_idx = self.bayer_unify(raw_img, bayer_type)
        raw_img, h_pad, w_pad = self.pad_to_mutiple(raw_img)
        raw_img_tensor = torch.tensor(raw_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        dm_img = self.rstcanet(raw_img_tensor)
        dm_img = dm_img[..., :-h_pad, :-w_pad]
        dm_img = dm_img.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1)
        return dm_img[h_idx[0]:h_idx[1], w_idx[0]:w_idx[1]]

    def bayer_unify(self, raw_img, bayer_type):
        """
        padding raw_img to make it get 'RGGB' bayer pattern for RSTCANet
        Return:
            raw_img: shape [H + (0,2), W + (0,2), 1]
            h_pad: index of true image in height
            w_pad: index of true image in width
        """
        if bayer_type.lower() == 'rggb':
            return raw_img, (None, None), (None, None)
        elif bayer_type.lower() == 'grbg':
            return np.pad(raw_img, ((0, 0), (1, 1), (0, 0)), mode='reflect'), (None, None), (1, -1)
        elif bayer_type.lower() == 'gbrg':
            return np.pad(raw_img, ((1, 1), (0, 0), (0, 0)), mode='reflect'), (1, -1), (None, None)
        elif bayer_type.lower() == 'bggr':
            return np.pad(raw_img, ((1, 1), (1, 1), (0, 0)), mode='reflect'), (1, -1), (1, -1)

    def pad_to_mutiple(self, raw_img, multiple=64):
        h, w = raw_img.shape[:2]
        h_pad, w_pad = -h % multiple, -w % multiple
        return np.pad(raw_img, ((0, h_pad), (0, w_pad), (0, 0)), mode='reflect'), h_pad, w_pad


class QuadRawConverter(RawConverter):
    def __init__(self, bayer_type: str='grbg', dm_type: str='dl_engine', ckpt_path: str=None, device: str='cuda') -> None:
        super().__init__(bayer_type)
        raise NotImplementedError
        # 지금 이 부분은 구현하지 않아도 될 것 같음


""" ───────────────────────────────────────────────────────
                            Model
─────────────────────────────────────────────────────── """
class RSTCANet(nn.Module):
    '''RSTCANet'''
    def __init__(self, in_nc=1, out_nc=3, patch_size=2, nc=72, window_size=8, img_size=[64, 64],
                 num_heads=[6,6], depths = [6,6]):
        super(RSTCANet, self).__init__()

        m_pp = B.PixelUnShuffle(patch_size)
        m_le = B.LinearEmbedding(in_channels=in_nc*patch_size*patch_size, out_channels=nc)
        pos_drop = nn.Dropout(p=0.)
        self.head = B.sequential(m_pp, m_le, pos_drop)

        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]



        self.m_body = B.Body(patches_resolution, depths=depths, num_heads=num_heads, nc=nc, window_size=window_size)

        self.m_ipp = B.PatchUnEmbedding(nc)
        self.m_conv = B.conv(nc//patch_size//patch_size, nc, bias=True, mode='C')
        self.m_final_conv = B.conv(nc, out_nc, bias=True, mode='C')

    def forward(self, x0):
        '''
        encoder
        '''
        x1 = self.head(x0)
        x_size = (x0.shape[2]//2, x0.shape[3]//2)
        x = self.m_body(x1, x_size)
        x = self.m_ipp(x, x_size)
        x = self.m_conv(x)
        x = self.m_final_conv(x)

        return x


if __name__ == '__main__':
    bayer_path = 'scripts_data/mx/dm_folder/20240928_SNU_PureRaw2IdealRaw_sample/2_IdealRaw_output/12MP_IN/MFP30/0036_2_E3Q_IN_Real_240723_101752/LSCApplyGainTableOut_4080x3060_0_12b.raw'
    bayer_path = 'scripts_data/mx/dm_folder/20240928_SNU_PureRaw2IdealRaw_sample/2_IdealRaw_output/12MP_GT/MFP30/0036_2_E3Q_GT_240723_101752/LSCApplyGainTableOut_4080x3060_0_12b.raw'
    m = BayerRawConverter(dm_type='menon', ckpt_path='ckpt/demosaic/RSTCANet/RSTCANet_B.pth')
    rgb_img = m.convert(m.read_raw(bayer_path))
    print(rgb_img.shape)
    rgb_img = (rgb_img * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb_img)
    rgb_img.save('tmp/rstcanet_converted.png')
    print('RSTCANet image saved: tmp/rstcanet_converted.png')
