import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, Model, losses, optimizers
from tensorflow.keras.losses import Loss
import numpy as np


angularDim=[7,7]
spatialDim=[128,128]
myDtype = tf.float32
myIntDtype = tf.int32


@tf.function()
def center_render_loss(center,center_depth, lf_lam):
    centers = render_center(center_depth, lf_lam)
    center_integral = tf.reduce_mean(centers, axis=(3, 4))
    loss = photometric_loss_center(center,center_integral)
    return loss, centers

@tf.function(experimental_relax_shapes=True)
def photometric_loss_center(y_true, y_pred):
    y_true = lf_zero_to_one(y_true)
    y_pred = lf_zero_to_one(y_pred)
    MAE = tf.reduce_mean(losses.MAE(y_true,y_pred),axis=(1,2))
    return tf.reduce_sum(tf.constant(.85) * (tf.constant(1.) - tf.image.ssim(y_true, y_pred, max_val=1.0)/tf.constant(2.)) \
    + tf.constant(0.15) * MAE)

@tf.function()
def photometric_loss(y_true, y_pred):
    y_true = lf_zero_to_one(y_true)
    y_pred = lf_zero_to_one(y_pred)
    MAE = tf.reduce_mean(losses.MAE(y_true,y_pred),axis=(1,2))
    y_true = tf.transpose(y_true,[0,3,4,1,2,5])
    y_pred = tf.transpose(y_pred,[0,3,4,1,2,5])
    ssim = tf.image.ssim(y_pred, y_true, max_val=1.0)
    return tf.reduce_sum(tf.constant(.85) * (tf.constant(1.) - ssim)/tf.constant(2.) + tf.constant(0.15) * MAE)

@tf.function(experimental_relax_shapes=True)
def PhotometricLoss(centers,lam_lf,LF_true):
    center = LF_true[:,:,:,3,3,:3]
    center = tf.expand_dims(tf.expand_dims(center,3),4)
    center = tf.tile(center,[1,1,1,7,7,1])
    part1 = photometric_loss(LF_true, lam_lf)
    part2 = photometric_loss(center,centers)
    return part1 + part2

@tf.function()
def depth_consistency_loss(center_d,ray_depth):
    ray_depth2 = depth_rendering_lambertian(center_d,ray_depth)
    return tf.reduce_sum(tf.reduce_mean(tf.abs(tf.subtract(ray_depth2,ray_depth)),axis=(1,2)))

@tf.function(experimental_relax_shapes=True)
def dof_metrics(y_true, y_pred):
    y_true = lf_zero_to_one(y_true)
    y_pred = lf_zero_to_one(y_pred)
    dof_true = tf.reduce_mean(y_true, axis=(3, 4))
    dof_pred = tf.reduce_mean(y_pred, axis=(3, 4))
    psnr = tf.image.psnr(dof_pred, dof_true, max_val=1.0)
    ssim = tf.image.ssim(dof_pred, dof_true, max_val=1.0)
    return tf.reduce_mean(ssim), tf.reduce_mean(psnr)


@tf.function(experimental_relax_shapes=True)
def dof_loss(y_true, y_pred):
    y_true = lf_zero_to_one(y_true)
    y_pred = lf_zero_to_one(y_pred)
    dof_true = tf.reduce_mean(y_true, axis=(3, 4))
    dof_pred = tf.reduce_mean(y_pred, axis=(3, 4))
    MAE = tf.reduce_mean(losses.MAE(dof_true,dof_pred),axis=(1,2))
    ssim = tf.image.ssim(dof_true,dof_pred,max_val=1.0)
    return tf.reduce_sum(tf.constant(.85) * (tf.constant(1.) - ssim)/tf.constant(2.) + tf.constant(0.15) * MAE)

@tf.function(experimental_relax_shapes=True)
def ssim_metric(y_true,y_pred):
    y_true = lf_zero_to_one(y_true)
    y_true = tf.image.convert_image_dtype(y_true,dtype=tf.uint8)
    y_pred = lf_zero_to_one(y_pred)
    y_pred = tf.image.convert_image_dtype(y_pred,dtype=tf.uint8)
    ssim = tf.image.ssim(y_true,y_pred,max_val=255)
    return ssim

@tf.function(experimental_relax_shapes=True)
def psnr_metric(y_true,y_pred):
    y_true = lf_zero_to_one(y_true)
    y_true = tf.image.convert_image_dtype(y_true,dtype=tf.uint8)
    y_pred = lf_zero_to_one(y_pred)
    y_pred = tf.image.convert_image_dtype(y_pred,dtype=tf.uint8)
    psnr = tf.image.psnr(y_true,y_pred,max_val=255)
    psnr = tf.clip_by_value(psnr,0,clip_value_max=100.0)
    return psnr

@tf.function(experimental_relax_shapes=True)
def center_metrics(y_true,y_pred):
    y_true = lf_zero_to_one(y_true)
#     y_true = tf.image.convert_image_dtype(y_true,dtype=tf.uint8)
    y_pred = lf_zero_to_one(y_pred)
#     y_pred = tf.image.convert_image_dtype(y_pred,dtype=tf.uint8)
    psnr = tf.image.psnr(y_true,y_pred,max_val=1.0)
    psnr = tf.clip_by_value(psnr,0,clip_value_max=100.0)
    ssim = tf.image.ssim(y_true,y_pred,max_val=1.0)
    return tf.reduce_mean(psnr),tf.reduce_mean(ssim)


@tf.function(experimental_relax_shapes=True)
def lf_metrics(y_true, y_pred):
    y_true = lf_zero_to_one(y_true)
    y_pred = lf_zero_to_one(y_pred)
    y_true = tf.transpose(y_true,[0,3,4,1,2,5])
    y_pred = tf.transpose(y_pred,[0,3,4,1,2,5])
    psnr = tf.image.psnr(y_pred, y_true, max_val=1.0)

    psnr = tf.clip_by_value(psnr,0,clip_value_max=100.0)
    psnrSum = tf.reduce_sum(psnr,axis=(1,2))

    ssim = tf.image.ssim(y_pred, y_true, max_val=1.0)
    ssimSum = tf.reduce_sum(ssim,axis=(1,2))    

    return tf.reduce_mean(ssimSum/(angularDim[0]*angularDim[1])), tf.reduce_mean(psnrSum/(angularDim[0]*angularDim[1]))


@tf.function(experimental_compile=True)
def depth_rendering_center(LF, ray_depths):
    b_sz = tf.shape(ray_depths)[0]
    y_sz = tf.shape(ray_depths)[1]
    x_sz = tf.shape(ray_depths)[2]

    #create and reparameterize light field grid
    b_vals = tf.cast(tf.range(b_sz),myDtype)
    y_vals = tf.cast(tf.range(y_sz),myDtype)
    x_vals = tf.cast(tf.range(x_sz),myDtype)

    b, y, x = tf.meshgrid(b_vals, y_vals, x_vals, indexing='ij')

    uList = [tf.multiply(tf.ones([b_sz,y_sz,x_sz]),tf.cast(tf.subtract(u0,angularDim[0]//2),dtype=myDtype)) for u0 in range(0,7)]
    vList = [tf.multiply(tf.ones([b_sz,y_sz,x_sz]),tf.cast(tf.subtract(v0,angularDim[0]//2),dtype=myDtype)) for v0 in range(0,7)]


    #warp coordinates by ray depths
    y_t_list = [tf.math.subtract(y, tf.math.multiply(v, ray_depths)) for v in vList]
    x_t_list = [tf.math.subtract(x, tf.math.multiply(u, ray_depths)) for u in uList]
    v_r = tf.zeros_like(b)
    u_r = tf.zeros_like(b)

    #indices for linear interpolation
    b_1 = tf.cast(b,myIntDtype)
    y_1_list = [tf.cast(tf.floor(y_t),myIntDtype) for y_t in y_t_list]
    y_2_list = [tf.math.add(y_1, 1) for y_1 in y_1_list]
    x_1_list = [tf.cast(tf.floor(x_t),myIntDtype) for x_t in x_t_list]
    x_2_list = [tf.math.add(x_1,1) for x_1 in x_1_list]
    v_1 = tf.cast(v_r,myIntDtype)
    u_1 = tf.cast(u_r,myIntDtype)

    y_1_list = [tf.clip_by_value(y_1, 0, y_sz-1) for y_1 in y_1_list]
    y_2_list = [tf.clip_by_value(y_2, 0, y_sz-1) for y_2 in y_2_list]
    x_1_list = [tf.clip_by_value(x_1, 0, x_sz-1) for x_1 in x_1_list]
    x_2_list = [tf.clip_by_value(x_2, 0, x_sz-1) for x_2 in x_2_list]

    #assemble interpolation indices
    interp_pts_1_List = [[tf.stack([b_1, y_1, x_1], -1) for x_1 in x_1_list] for y_1 in y_1_list]
    interp_pts_2_List = [[tf.stack([b_1, y_2, x_1], -1) for x_1 in x_1_list] for y_2 in y_2_list]
    interp_pts_3_List = [[tf.stack([b_1, y_1, x_2], -1) for x_2 in x_2_list] for y_1 in y_1_list]
    interp_pts_4_List = [[tf.stack([b_1, y_2, x_2], -1) for x_2 in x_2_list] for y_2 in y_2_list]
    lf_1_list = [[tf.gather_nd(LF[:,:,:,j,i], interp_pts_1_List[j][i]) for i in range(0,7)] for j in range(0,7)]
    lf_2_list = [[tf.gather_nd(LF[:,:,:,j,i], interp_pts_2_List[j][i]) for i in range(0,7)] for j in range(0,7)]
    lf_3_list = [[tf.gather_nd(LF[:,:,:,j,i], interp_pts_3_List[j][i]) for i in range(0,7)] for j in range(0,7)]
    lf_4_list = [[tf.gather_nd(LF[:,:,:,j,i], interp_pts_4_List[j][i]) for i in range(0,7)] for j in range(0,7)]
    
    y_1_f_list = [tf.cast(y_1,myDtype) for y_1 in y_1_list]
    x_1_f_list = [tf.cast(x_1,myDtype) for x_1 in x_1_list]
    
    d_y_1_list = [tf.math.subtract(1.0, tf.math.subtract(y_t_list[i], y_1_f_list[i])) for i in range(0,7)]
    d_y_2_list = [tf.math.subtract(1.0, d_y_1) for d_y_1 in d_y_1_list]
    d_x_1_list = [tf.math.subtract(1.0, tf.math.subtract(x_t_list[i], x_1_f_list[i])) for i in range(0,7)]
    d_x_2_list = [tf.math.subtract(1.0, d_x_1) for d_x_1 in d_x_1_list]
    
    w1_list = [[tf.math.multiply(d_y_1, d_x_1) for d_x_1 in d_x_1_list] for d_y_1 in d_y_1_list]
    w2_list = [[tf.math.multiply(d_y_2, d_x_1) for d_x_1 in d_x_1_list] for d_y_2 in d_y_2_list]
    w3_list = [[tf.math.multiply(d_y_1, d_x_2) for d_x_2 in d_x_2_list] for d_y_1 in d_y_1_list]
    w4_list = [[tf.math.multiply(d_y_2, d_x_2) for d_x_2 in d_x_2_list] for d_y_2 in d_y_2_list]
    center_list = [[tf.add_n(
        [w1_list[i][j]*tf.cast(lf_1_list[i][j],myDtype),
         w2_list[i][j]*tf.cast(lf_2_list[i][j],myDtype),
         w3_list[i][j]*tf.cast(lf_3_list[i][j],myDtype),
         w4_list[i][j]*tf.cast(lf_4_list[i][j],myDtype)]
         ) for i in range(0,7)] for j in range(0,7)]

    centers_list = [tf.stack(center_list[i], -1) for i in range(0,7)]
    centers = tf.stack(centers_list,-1)
                    
    return centers

@tf.function
def render_center(center_depth, lf_lam):
    all_centeres_r = depth_rendering_center(lf_lam[:,:,:,:,:,0], center_depth)
    all_centeres_g = depth_rendering_center(lf_lam[:,:,:,:,:,1], center_depth)
    all_centeres_b = depth_rendering_center(lf_lam[:,:,:,:,:,2], center_depth)
    all_centers = tf.stack([all_centeres_r,all_centeres_g,all_centeres_b], axis=-1)
    return all_centers

@tf.function
def lambertian_lf_gen(ray_depth, center):
    lf_lambertian_r = depth_rendering_lambertian(center[:,:,:,0],ray_depth)
    lf_lambertian_g = depth_rendering_lambertian(center[:,:,:,1],ray_depth)
    lf_lambertian_b = depth_rendering_lambertian(center[:,:,:,2],ray_depth)
    lf_lambertian = tf.stack([lf_lambertian_r,lf_lambertian_g,lf_lambertian_b], axis=5)
    return lf_lambertian

@tf.function(experimental_relax_shapes=True)
def lf_zero_to_one(lf):
    return tf.math.add(tf.math.divide(lf,tf.constant(2.)),tf.constant(0.5))

@tf.function
def depth_rendering_lambertian(central, ray_depths):
    b_sz = tf.shape(ray_depths)[0]
    y_sz = tf.shape(ray_depths)[1]
    x_sz = tf.shape(ray_depths)[2]
    u_sz = tf.shape(ray_depths)[3]
    v_sz = tf.shape(ray_depths)[4]
    
    central = tf.expand_dims(tf.expand_dims(central, 3), 4)
                                            
    #create and reparameterize light field grid
    b_vals = tf.cast(tf.range(b_sz),myDtype)
    v_vals = tf.math.subtract(tf.cast(tf.range(v_sz),myDtype), tf.cast(v_sz,myDtype)//2.0)
    u_vals = tf.math.subtract(tf.cast(tf.range(u_sz),myDtype), tf.cast(u_sz,myDtype)//2.0)
    y_vals = tf.cast(tf.range(y_sz),myDtype)
    x_vals = tf.cast(tf.range(x_sz),myDtype)

    b, y, x, v, u = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')
            
    #warp coordinates by ray depths
    y_t = tf.math.add(y, tf.math.multiply(v, ray_depths))
    x_t = tf.math.add(x, tf.math.multiply(u, ray_depths))
    
    v_r = tf.zeros_like(b)
    u_r = tf.zeros_like(b)
    
    #indices for linear interpolation
    b_1 = tf.cast(b,myIntDtype)
    y_1 = tf.cast(tf.floor(y_t),myIntDtype)
    y_2 = tf.math.add(y_1, 1)
    x_1 = tf.cast(tf.floor(x_t),myIntDtype)
    x_2 = tf.math.add(x_1,1)
    v_1 = tf.cast(v_r,myIntDtype)
    u_1 = tf.cast(u_r,myIntDtype)
    
    y_1 = tf.clip_by_value(y_1, 0, y_sz-1)
    y_2 = tf.clip_by_value(y_2, 0, y_sz-1)
    x_1 = tf.clip_by_value(x_1, 0, x_sz-1)
    x_2 = tf.clip_by_value(x_2, 0, x_sz-1)
    
    #assemble interpolation indices
    interp_pts_1 = tf.stack([b_1, y_1, x_1, v_1, u_1], -1)
    interp_pts_2 = tf.stack([b_1, y_2, x_1, v_1, u_1], -1)
    interp_pts_3 = tf.stack([b_1, y_1, x_2, v_1, u_1], -1)
    interp_pts_4 = tf.stack([b_1, y_2, x_2, v_1, u_1], -1)
    
    #gather light fields to be interpolated
    lf_1 = tf.gather_nd(central, interp_pts_1)
    lf_2 = tf.gather_nd(central, interp_pts_2)
    lf_3 = tf.gather_nd(central, interp_pts_3)
    lf_4 = tf.gather_nd(central, interp_pts_4)
    
    #calculate interpolation weights
    y_1_f = tf.cast(y_1,myDtype)
    x_1_f = tf.cast(x_1,myDtype)
    d_y_1 = tf.math.subtract(1.0, tf.math.subtract(y_t, y_1_f))
    d_y_2 = tf.math.subtract(1.0, d_y_1)
    d_x_1 = tf.math.subtract(1.0, tf.math.subtract(x_t, x_1_f))
    d_x_2 = tf.math.subtract(1.0, d_x_1)
    
    w1 = tf.math.multiply(d_y_1, d_x_1)
    w2 = tf.math.multiply(d_y_2, d_x_1)
    w3 = tf.math.multiply(d_y_1, d_x_2)
    w4 = tf.math.multiply(d_y_2, d_x_2)
    
    lf = tf.add_n([w1*tf.cast(lf_1,myDtype), w2*tf.cast(lf_2,myDtype), w3*tf.cast(lf_3,myDtype), w4*tf.cast(lf_4,myDtype)])
                    
    return lf

