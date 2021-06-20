#!/usr/bin/env python
# coding: utf-8

# # Code

# In[2]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import pymcfost as mcfost
mcfost.__version__
#%matplotlib inline

#p_dirin -> directory with mcfost files .para e RT.fits.gz
#sim_para -> dict with parameters that need to be added to the header
#function that generates the post pymcfost image with added information in the header
def generateImage(p_dirin,
                  p_bmaj,
                  p_bmin,
                  p_bpa = 0,
                  p_dirout='./',
                  fileout='ris2.fits',
                  p_Jy=True,
                  p_perbeam=True,
                  p_i=0,
                  sim_para={},
                  ax=None
                 ):
    
    image = mcfost.Image(p_dirin)
    pltimage = image.plot(i=p_i, bpa=p_bpa, bmaj=p_bmaj,
                          bmin=p_bmin,scale='lin',Jy=p_Jy, per_beam=p_perbeam,
                          no_ylabel=True,no_xlabel=True,
                          plot_stars=False,limits=[0.8,-0.8,-0.8,0.8], ax=ax, title=fileout)
    header = image.header
    
    #manually updating header.BUNIT
    bunit_1 = 'W.m-2'
    bunit_2 = '.pixel-1'
    if p_Jy:
        bunit_1 = 'Jy'
    if p_perbeam:
        bunit_2 = '.beam-1'
    sim_para['BUNIT'] = bunit_1 + bunit_2
    
    #adding simulation parameters to the sim_para
    
    #star properties
    nstars = image.P.simu.n_stars
    sim_para['NSTARS'] = nstars
    for i in range(0, nstars):
        sim_para['TSTAR' + str(i)] = image.P.stars[i].Teff, 'K'
        sim_para['MSTAR' + str(i)] = image.P.stars[i].M, 'M_sun'
        sim_para['RSTAR' + str(i)] = image.P.stars[i].R, 'R_sun'
        
    #beam informations
    sim_para['BMAJ'] = p_bmaj
    sim_para['BMIN'] = p_bmin
    sim_para['BPA'] = p_bpa, 'deg'
        
    #position angle
    sim_para['DISKPA'] = image.P.map.PA, 'deg'
    
    #inclination
    imin = np.deg2rad( image.P.map.RT_imin)
    imax = np.deg2rad(image.P.map.RT_imax)
    ni = image.P.map.RT_ntheta
    cos_i = np.linspace(np.cos(imin), np.cos(imax), ni)
    sim_para['INCL']  = (np.rad2deg(np.arccos(cos_i[p_i])), 'deg')
    
    #distance
    sim_para['DISTANCE'] = image.P.map.distance, 'pc'
    
    #getting wave 
    sim_para['WAVE'] = header['WAVE']
    
    #calling function for final computations
    set_computable_features(sim_para)
    
    #adding info in sim_para dict
    for key in sim_para:
        header[key] = sim_para[key]
        
    #saving fits file
    image.writeto(filename=p_dirout+'/' + fileout)
    
    #returning info about additional inclinations
    #-1 is returned if there are no more inclinations available
    if p_i < ni-1 and p_i>0:
        next_i = p_i+1
    else:
        next_i = -1
    return next_i


# In[3]:


#function that extracts informations from image.log file
def get_imagelog_info(sim_para, p_dir):
    logf = open(p_dir+'/image.log')
    data = logf.read()
    
    #read gass mass and dust mass
    i_mg = data.find('gas mass')
    str_mg = data[i_mg:i_mg+data[i_mg:].find('\n')].split(' ')
    sim_para['MGAS']  = (float(str_mg[-2]), str_mg[-1])
    if str_mg[-1]!='Msun' :
        print('Error: gas mass not in M_sun units')
    i_md = data.find('dust mass')
    str_md = data[i_md:i_md+data[i_md:].find('\n')].split(' ')
    sim_para['MDUST']  = (float(str_md[-2]), str_md[-1])
    if str_md[-1]!='Msun' :
        print('Error: dust mass not in M_sun units')
        
    #read planet mass and orbital radius
    i_sink = []
    i_sink.append(data.find('Sink'))
    i_sink.append(i_sink[-1] + data[i_sink[-1]+1:].find('Sink'))
    while i_sink[-1]-i_sink[-2]!= 0:
        i_sink.append(i_sink[-1] + 1+ data[i_sink[-1]+1:].find('Sink'))
    i_sink.pop(-1)
    i_sink.pop(0)
    i_sink.append(i_sink[-1]+200)
    str_sink = [data[i_sink[i]:i_sink[i+1]] for i in range(0, len(i_sink)-1)]

    sink_index = 0
    for info in str_sink:
        if info.find('distance') > 0:
            info = " ".join(info.split())
            i_m = info.find('M')
            str_m = info[i_m:i_m+1+info[i_m+1:].find('M')+2]
            sim_para['MPLANET'+str(sink_index)] = float(str_m.split('=')[-1].split(' ')[-2]), str_m.split('=')[-1].split(' ')[-1]
            i_r = info.find('distance')
            str_r = info[i_r:i_r+1+info[i_r+1:].find('au')+2]
            sim_para['RORBP'+str(sink_index)] = float(str_r.split('=')[-1].split(' ')[-2]), str_r.split('=')[-1].split(' ')[-1]
            sink_index+=1
    sim_para['NPLANETS'] = sink_index


# In[4]:


import os
import re
import hashlib
from pathlib import Path

def read_dirs_doimages(parent_dir='./', beam_dims=[0.1]):
    
    #setting global parameters, values for  dstau simulations
    sim_para = {
        'RIN': (10, 'au, initial disk inner radius' ),
        'ROUT': (100, 'au, initial disk outer radius'),
        'RC': (70, 'au, radius of exponential taper'),
        'HRIN': (0.6,'aspect ratio at R_in'),
        'ALPHASS': (0.005, 'Shakura-Sunyaev viscosity'),
        'RHOG': (1, 'g/cm^3, intrinsic grain density'),
        'PINDEX': (1, 'power law index for gas surface density profile'),
        'QINDEX': (0.25, 'power law index for sound speed profile'),
    }
    
    current_dir = parent_dir
    key = 0
    out_dir = parent_dir + 'processed_fits/'
    out_dir_content = ''
    
    #setting up plotting
    key_plots = 0
    curr_sub = 0
    #please don't set rows or cols to 1!
    plot_rows = 10
    plot_cols = 10
    fig, axs = plt.subplots(plot_rows, plot_cols)
    fig.set_size_inches(5*plot_cols,5*plot_rows)
    
    #getting dir content
    dir_cont = os.listdir(parent_dir)
    
    #creating dir for the fits file that will be produced
    if 'processed_fits' not in dir_cont:
        os.mkdir(parent_dir + 'processed_fits')
        out_dir = parent_dir + 'processed_fits/'
        Path(out_dir+'source_hashes.sha1').touch()
    else:
        #if the output dir already exists check for last key and update
        #retrieving output dir content 
        out_dir_content = os.listdir(out_dir)
        
        #if .fits exists get max key already present and restart from there
        fits_sim_mod = re.compile("^\d{6}MP\d+_time\d+gd\d+W\d+B\d+.fits$")
        keys = [int(filename[:6]) for filename in out_dir_content if fits_sim_mod.match(filename)]
        if keys != []:
            key = max(keys) + 1
        
        #if _images.png exists get max key_plots already present and restart from there
        im_mod = re.compile("^\d{3}_images.png$")
        keys = [int(filename[:3]) for filename in out_dir_content if im_mod.match(filename)]
        if keys != []:
            key_plots = max(keys) + 1
            
        if 'source_hashes.sha1' not in out_dir_content:
            Path(out_dir+'source_hashes.sha1').touch()
            
    #cleaning dir content
    dir_mod = re.compile("^MP[0-9]+_time[0-9]+$")
    subdirs1 = [sd for sd in dir_cont if dir_mod.match(sd)]
    
    #cicle for /MP#_time#/ 
    for sub_dir1 in subdirs1:
        
        #getting time of simulation
        sim_para['TIME'] = int(sub_dir1.split('time')[-1]), 'planet orbits'
        
        #getting subdirs with the correct format
        dir_cont2 = os.listdir(parent_dir+sub_dir1)
        dir_mod2 = re.compile("^gd[0-9]+$")
        subdirs2 = [sd for sd in dir_cont2 if dir_mod2.match(sd)]
        
        #cicle for /MP#_time#/gd#/
        for sub_dir2 in subdirs2:
            
            #from now on using sim_para_upd to avoid
            #having to delete info that are not overrided
            sim_para_upd = sim_para 
            
            #check if there is image.log file otherwise skip
            dir_content = os.listdir(parent_dir+sub_dir1+'/'+sub_dir2+'/')
                
            if 'image.log' not in dir_content:
                print('image.log file not found in ' + sub_dir1+'/'+sub_dir2+ ': skipping')
                continue
                
            #this directory contains image.log and data_### dirs
            #updating sim_para with info contained in image.log
            get_imagelog_info(sim_para=sim_para_upd, p_dir=parent_dir+sub_dir1+'/'+sub_dir2)
            
            #searching for data_### dirs
            #getting subdirs
            dir_cont3 = os.listdir(parent_dir+sub_dir1+'/'+sub_dir2)
            dir_mod3 = re.compile("^data_[0-9]+$")
            subdirs3 = [sd for sd in dir_cont3 if dir_mod3.match(sd)]
            
            #cicle for /MP#_time#/gd#/data_#
            for sub_dir3 in subdirs3:
                
                current_dir = parent_dir + sub_dir1 + '/' + sub_dir2 + '/' + sub_dir3 + '/'
                
                for beam_dim in beam_dims:
                    
                    #select inclination
                    #start from i=0, generate_image return next_i if there is more than one inclination
                    p_i = 0
                    
                    while p_i >= 0:
                        #check if a new image for plots needs to be created
                        if curr_sub == plot_rows*plot_cols:
                                #save plot
                                plt.tight_layout()
                                print('generating image ' + str(key_plots) + ' ...', end='')
                                plt.savefig(out_dir + str(key_plots).zfill(3) + '_images.png')
                                plt.close()
                                print('done')

                                #generating new image
                                curr_sub = 0
                                fig, axs = plt.subplots(plot_rows, plot_cols)
                                fig.set_size_inches(10*plot_cols,10*plot_rows)
                                key_plots+=1

                        #generate file name
                        key_str = str(key).zfill(6)
                        wave_str = sub_dir3.split('data_')[-1]
                        out_filename = key_str + sub_dir1 + sub_dir2 + 'W' + wave_str + 'B' + str(beam_dim).split('.')[-1][:3] + '.fits'
                        print(key_str + ' - generating ' + out_filename[6:] + '  \t...', end='')
                      
                        #check if RT.fits.gz has alredy been processed
                        info_to_hash = 'bd' + str(beam_dim) +'ii'+ str(p_i) + 'fn' + out_filename[6:] + 'pi'+str(p_i)
                        hashes_f = open(out_dir + 'source_hashes.sha1')
                        hashes = [line.split('-')[-1].strip() for line in hashes_f.read().split('\n')]
                        RTfile = open(current_dir + 'RT.fits.gz', 'rb')
                        xHash = hashlib.sha1((hashlib.sha1(RTfile.read()).hexdigest()+info_to_hash).encode()).hexdigest()
                        RTfile.close() 
                        hashes_f.close()

                        if xHash not in hashes:
                            p_i = generateImage(current_dir, p_bmaj=beam_dim, 
                                          p_bmin=beam_dim, p_dirout=out_dir, 
                                          fileout=out_filename, sim_para=sim_para_upd, ax=axs[int(curr_sub/plot_cols), curr_sub%plot_cols])
                            key+=1
                            hashes_f = open(out_dir + 'source_hashes.sha1', 'a')
                            hashes_f.write(key_str + ' - ' + xHash + '\n')
                            hashes_f.close()

                            print('done')
                            curr_sub+=1

                        else:
                            print('exists, skipped')
                            break
                        
    print('generating image ' + str(key_plots) + ' ...', end='')
    plt.tight_layout()
    plt.savefig(out_dir + str(key_plots).zfill(3) + '_images.png') 
    print('done')
    plt.close()


# In[5]:


from astropy import constants as const

def HR_profile(r, hrin=0.06, rin=10, q=0.25):
    return hrin*(r/rin)**(0.5-q)

def cs_profile(r, hrin=0.06, rin=10, q=0.25, M_st=0.83):
    r_si = r*const.au.value
    v_k  = np.sqrt(float(const.GM_sun.value)*M_st/r_si)
    return HR_profile(r=r, hrin=hrin, rin=rin, q=q)*v_k

def T_profile(r, hrin=0.06, rin=10, q=0.25, M_st=0.83, rho_g=1, s=1 ):
    r_si = r*const.au.value
    return rho_g*4*np.pi*(s*(10**6))**3/(3*const.k_B.value*1000)*(cs_profile(r_si, hrin=hrin, rin=rin, q=q, M_st=M_st)**2)


# In[6]:


def set_computable_features(sim_para):
    
    #computing features
    s = sim_para['WAVE']/(2*np.pi)
    rp = sim_para['RORBP0'][0]
    hrp = HR_profile(rp, hrin=sim_para['HRIN'][0], rin=sim_para['RIN'][0], q=sim_para['QINDEX'][0])
    Trp = T_profile(rp, hrin=sim_para['HRIN'][0], rin=sim_para['RIN'][0], q=sim_para['QINDEX'][0], M_st=sim_para['MSTAR0'][0], rho_g=sim_para['RHOG'][0], s=s)

    #adding them to sim_para
    sim_para['GRSIZE'] = (s, 'grain size')
    sim_para['HR'] = (hrp, 'aspect ratio at orbital radius')
    sim_para['TEMP'] = (Trp, 'temperature at orbital radius')
    

#generate list with beam dimensions
def get_beam_dims(n, dmin=0.01, dmax=1):
    return np.logspace(np.log10(dmin), np.log10(dmax), n)




