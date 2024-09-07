## This script includes the inference (reverse sampling) utilities for apply the diffusion model
import torch

class Inferencer(object):
    def __init__(self, 
                 nets, 
                 config):
        self.nets = nets
        self.config = config

    def run_once(self, batch):
        self.nets.eval()
        with torch.no_grad():
            if self.distributed:
                ensembel,mole_padding,prot_padding = self.nets.module.eval_once(batch, self.sampler, 
                                                                            T = self.config.TRAIN['max_reverse_diffusion_time'],
                                                                             stochastic=self.config.TRAIN['stochastic_reverse_sampling'],
                                                                             record_intermediate = self.config.TRAIN['record_intermediate'])
            else:
                ensembel,mole_padding,prot_padding = self.nets.eval_once(batch, self.sampler, 
                                                                      T = self.config.TRAIN['max_reverse_diffusion_time'],
                                                                      stochastic=self.config.TRAIN['stochastic_reverse_sampling'],
                                                                      record_intermediate = self.config.TRAIN['record_intermediate'])
            label = self.get_label_coord(batch)
            coord = ensembel[-1]
            mole_rmsd,prot_rmsd = self.rmsd(coord, label, mole_padding, prot_padding)
            #calculate the original rmsd
            orig_coord = torch.cat([batch['net_input']['mol_src_coord'],batch['net_input']['pocket_src_coord']],dim=1)
            mole_rmsd_ori,prot_rmsd_ori = self.rmsd(orig_coord,label,mole_padding, prot_padding)
            if self.config.TRAIN['record_intermediate'] and self._on_main_rank():
                self.record_intermediate(ensembel,
                                         label,
                                         mole_padding, 
                                         prot_padding, 
                                         batch['net_input']['mol_tokens'],
                                         batch['net_input']['pocket_tokens'],
                                         batch['pocket_name'])
    
if __name__ == "__main__":
    