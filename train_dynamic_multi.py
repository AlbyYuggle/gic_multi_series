# coding=utf-8
import sys
import torch
import taichi as ti
import torch.nn as nn
import time, os, json
from tqdm import tqdm, trange
from train_gs import training
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render
from scene import Scene, DeformModel
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from simulator import MPMSimulator, Estimator
from train_gs_fixed_pcd import train_gs_with_fixed_pcd, assign_gs_to_pcd
from utils.system_utils import check_gs_model, draw_curve, write_particles
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import numpy as np

image_scale = 1.0


def prepare_gt(dataset: ModelParams, iteration: int, pipeline: PipelineParams, phys_args):
    gts = []
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[image_scale])
    deform = DeformModel(dataset)
    deform.load_weights(dataset.model_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras(scale=image_scale)
    fids = torch.unique(torch.stack([view.fid for view in views]))
    xyz_canonical = gaussians.get_xyz.detach()
    opacitiy = gaussians.get_opacity.squeeze()
    grid_size = phys_args.density_grid_size
    density_min_th = phys_args.density_min_th
    density_max_th = phys_args.density_max_th
    num_iter = 4 if phys_args.random_sample else 5
    filling_grid_size = grid_size / 2 ** 5
    opacity_threshold = phys_args.opacity_threshold
    with torch.no_grad():
        for idx, fid in enumerate(tqdm(fids, desc="Filling progress")):
            if getattr(phys_args, "n_frames", None) and idx >= phys_args.n_frames:
                break
            time_input = fid.unsqueeze(0).expand(1, -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
            xyzt = xyz_canonical + d_xyz
            # xyzt = xyzt[opacitiy > opacity_threshold]
            bbox_mins = xyzt[opacitiy > opacity_threshold].min(dim=0)[0] - grid_size
            bbox_maxs = xyzt[opacitiy > opacity_threshold].max(dim=0)[0] + grid_size
            bbox_bounds = bbox_maxs - bbox_mins
            volume_size = torch.round(bbox_bounds / filling_grid_size).to(torch.int64) + 1
            grid_ids = [torch.arange(size) for size in volume_size]
            grid_coords = torch.stack(torch.meshgrid(*grid_ids, indexing='ij'), dim=-1).reshape(-1, 3) * filling_grid_size
            grid_coords = grid_coords.to(xyzt)
            init_inner_points = grid_coords + bbox_mins.reshape(1, 3)
            curr_views = [view for view in views if view.fid == fid]
            for viewpoint_cam in tqdm(curr_views, desc="Rendering progress"):
                results = render(viewpoint_cam, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                depth = results["depth"][0]
                render_mask = torch.logical_and(results["render"].sum(0) != 0,
                                                viewpoint_cam.original_image.sum(0) != 0)
                pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(init_inner_points)
                # remove points that are outside the image space
                in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
                init_inner_points = init_inner_points[in_mask]
                pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
                # remove points that the projected pixels are outside the object mask
                pix_mask = render_mask[pix_h, pix_w]
                init_inner_points = init_inner_points[pix_mask]
                pix_w, pix_h, pix_d = pix_w[pix_mask], pix_h[pix_mask], pix_d[pix_mask]
                # remove points whose depth values are smaller than those from depth map
                render_pix_d = depth[pix_h, pix_w]
                depth_mask = render_pix_d < pix_d
                init_inner_points = init_inner_points[depth_mask]
                # remove outliers in xyzt
                render_mask = results["render"].sum(0) > 1 / 255
                pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(xyzt)
                in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
                pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
                xyzt = xyzt[in_mask]
                pix_mask = render_mask[pix_h, pix_w]
                xyzt = xyzt[pix_mask]
            curr_grid_size = grid_size / 2
            volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
            bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
            bbox_bounds = bbox_maxs - bbox_mins
            density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(init_inner_points)
            ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            weight = torch.ones((1, 1, 3, 3, 3)).to(xyzt)
            weight = weight / weight.sum()
            for i in range(2, num_iter):
                curr_grid_size = grid_size / 2 ** i
                volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
                bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
                grid_xyz = torch.stack(torch.meshgrid(
                    torch.linspace(0, volume_size[0]-1, volume_size[0]),
                    torch.linspace(0, volume_size[1]-1, volume_size[1]),
                    torch.linspace(0, volume_size[2]-1, volume_size[2]),
                ), dim=-1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
                ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
                ids_norm = ids_norm[None].flip((-1,))
                density_volume = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)
                density_volume = torch.nn.functional.conv3d(density_volume, weight=weight, padding='same')[0, 0]
                density_volume[density_volume < 0.5] = 0.0
                ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
                ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
                bbox_bounds = bbox_maxs - bbox_mins
            for i in range(20):
                density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                density_volume[density_volume < 0.5] = 0.0
                ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
                ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            if phys_args.random_sample:
                density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                half_grid_xyz = torch.stack(torch.meshgrid(
                    torch.linspace(0, volume_size[0]-0.5, 2 * (volume_size[0]-1)),
                    torch.linspace(0, volume_size[1]-0.5, 2 * (volume_size[1]-1)),
                    torch.linspace(0, volume_size[2]-0.5, 2 * (volume_size[2]-1)),
                ), -1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
                ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
                ids_norm = ids_norm[None].flip((-1,))
                density_half_grid_xyz = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)[0, 0]
                half_grid_xyz = half_grid_xyz[density_half_grid_xyz > 0.5]
                delta = (torch.rand_like(half_grid_xyz) * curr_grid_size * 0.5).to(xyzt)
                particles = half_grid_xyz + delta
                ids_norm = (particles[None,None] - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
                ids_norm = ids_norm[None].flip((-1,))
                density_particles = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)[0, 0, 0, 0]
                sampled_pts = particles[density_particles > density_min_th]
                surface_pts = particles[(density_particles > density_min_th)*(density_particles < density_max_th)]
                gts.append(surface_pts)
                curr_grid_size = curr_grid_size / 2
                if fid == 0.0:
                    vol = sampled_pts
                    vol_densities = density_particles[density_particles > density_min_th]
                    vol_surface_mask = density_particles[density_particles > density_min_th] < density_max_th
                    vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]
                    write_particles(vol, 0, dataset.model_path, 'static')
            else:
                density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                internal_mask = density_volume >= density_min_th
                sampled_pts = torch.stack(torch.where(internal_mask), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
                density_volume_smoothed = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                surface_mask = (density_volume_smoothed > 0) * (density_volume_smoothed < density_max_th) * internal_mask
                surface_pts = torch.stack(torch.where(surface_mask == 1), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
                gts.append(surface_pts)
                if fid == 0.0:
                    vol = sampled_pts
                    vol_densities = density_volume[internal_mask]
                    vol_surface_mask = surface_mask[internal_mask]
                    vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]
                    write_particles(vol, 0, dataset.model_path, 'static')

        train_cams, test_cams, cameras_extent = scene.overwrite_alphas(pipeline, dataset, deform)
        cam_info = {
            "train_cams": train_cams,
            "test_cams": test_cams,
            "cameras_extent": cameras_extent,
        }
    return gts, vol, vol_densities, torch.tensor([curr_grid_size]), vol_surface, cam_info


def forward(estimator: Estimator, img_backward=True):
    dt = estimator.simulator.dt_ori[None]
    while True:
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize()
                estimator.simulator.set_dt(dt)
            x = estimator.forward(idx, img_backward)
        if not estimator.succeed():
            dt /= 2
            print('cfl condition dissatisfy, shrink dt {}, step cnt {}'.format(dt, estimator.simulator.n_substeps[None] * 2))
        else:
            break

def backward(estimator: Estimator):
    print('Geometry loss {}, image loss {}, step {}'.format(estimator.loss[None], estimator.image_loss, estimator.simulator.n_substeps[None]))
    max_f = estimator.max_f
    pbar = trange(max_f)
    pbar.set_description(f"[Backward]")

    estimator.loss.grad[None] = 1
    estimator.clear_grads()

    for ri in pbar:
        i = max_f - 1 - ri
        if i > 0:
            estimator.backward(i)
        else:
            pos_grad, velocity_grad, mu_grad, lam_grad, \
            yield_stress_grad, viscosity_grad, \
            friction_alpha_grad, cohesion_grad, rho_grad = estimator.backward(i)
            estimator.init_velocities.backward(retain_graph=True, gradient=velocity_grad)
            estimator.init_rhos.backward(retain_graph=True, gradient=rho_grad)
            estimator.init_pos.backward(retain_graph=True, gradient=pos_grad)
            estimator.init_mu.backward(retain_graph=True, gradient=mu_grad)
            estimator.init_lam.backward(retain_graph=True, gradient=lam_grad)
            estimator.yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
            estimator.plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
            estimator.friction_alpha.backward(retain_graph=True, gradient=friction_alpha_grad)
            estimator.cohesion.backward(gradient=cohesion_grad)

def train(estimator: Estimator, phys_args, max_f=None):
    losses = []
    estimated_params = []
    if estimator.stage[None] == Estimator.velocity_stage:
        iter_cnt = phys_args.vel_iter_cnt
    elif estimator.stage[None] == Estimator.physical_params_stage:
        iter_cnt = phys_args.iter_cnt

    if max_f is not None:
        estimator.max_f = max_f

    for stage, train_param in enumerate(zip([max_f], [iter_cnt])):
        max_f, iter_cnt = train_param
        if max_f is not None:
            estimator.max_f = max_f
        for i in range(iter_cnt):
            # 1. record current params
            d = {}
            param_groups = estimator.get_optimizer().param_groups
            report_msg = ''
            report_msg += f'iter {i}'
            report_msg += f'\nvelocity: {estimator.init_vel.cpu().detach().tolist()}'
            for params in param_groups:
                name = params['name']
                p = params['params'][0].detach().cpu()
                if name == 'Poisson ratio':
                    p = estimator.get_nu().detach().cpu()
                    report_msg += f'\n{name}: {p}'
                elif name in ['Youngs modulus', 'Yield stress', 'plastic viscosity', 'shear modulus', 'bulk modulus']:
                    p = 10**p
                    report_msg += f'\n{name}: {p}'
                #TODO: optimization
                if name != 'velocity':
                    d.update({name: p.item()})
                else:
                    d.update({name: p})
            print(report_msg)
            estimated_params.append(d)

            # 2. forward, backward, and update
            estimator.zero_grad()
            estimator.loss[None] = 0.0
            forward(estimator)
            losses.append(estimator.loss[None] + estimator.image_loss)
            backward(estimator)
            estimator.step(i)

            # 3. record loss and save best params
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
            print("Best params: ", best_params, 'in {} iteration'.format(min_idx))
            print("Min loss: {}".format(losses[min_idx]))

    if estimator.stage[None] == Estimator.velocity_stage and len(losses) > 0:
        min_idx = losses.index(min(losses))
        best_params = estimated_params[min_idx]
        estimator.init_vel = nn.Parameter(best_params['velocity'].to(estimator.device))

    return losses, estimated_params

def train_multi(estimators, phys_args_list, max_f=None):
    losses = []
    estimated_params = []
    if estimators[0].stage[None] == Estimator.velocity_stage:
        iter_cnt = phys_args_list[0].vel_iter_cnt
    elif estimators[0].stage[None] == Estimator.physical_params_stage:
        iter_cnt = phys_args_list[0].iter_cnt

    if max_f is not None:
        estimators[0].max_f = max_f

    for stage, train_param in enumerate(zip([max_f], [iter_cnt])):
        max_f, iter_cnt = train_param
        if max_f is not None:
            estimators[0].max_f = max_f
        for i in range(iter_cnt):
            # 1. record current params
            seq_no = torch.randint(0, len(estimators), (1,)).item()
            d = {}
            param_groups = estimators[seq_no].get_optimizer().param_groups
            report_msg = ''
            report_msg += f'iter {i}'
            report_msg += f'\nvelocity: {estimators[seq_no].init_vel.cpu().detach().tolist()}'
            for params in param_groups:
                name = params['name']
                p = params['params'][0].detach().cpu()
                if name == 'Poisson ratio':
                    p = estimators[seq_no].get_nu().detach().cpu()
                    report_msg += f'\n{name}: {p}'
                elif name in ['Youngs modulus', 'Yield stress', 'plastic viscosity', 'shear modulus', 'bulk modulus']:
                    p = 10**p
                    report_msg += f'\n{name}: {p}'
                #TODO: optimization
                if name != 'velocity':
                    d.update({name: p.item()})
                else:
                    d.update({name: p})
            print(report_msg)
            estimated_params.append(d)

            # 2. forward, backward, and update
            estimators[seq_no].zero_grad()
            estimators[seq_no].loss[None] = 0.0
            forward(estimators[seq_no])
            losses.append(estimators[seq_no].loss[None] + estimators[seq_no].image_loss)
            backward(estimators[seq_no])
            estimators[seq_no].step(i)

            for j in range(10):
                estimators[j].optimizer = estimators[seq_no].optimizer

            # 3. record loss and save best params
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
            print("Best params: ", best_params, 'in {} iteration'.format(min_idx))
            print("Min loss: {}".format(losses[min_idx]))

    if estimator.stage[None] == Estimator.velocity_stage and len(losses) > 0:
        min_idx = losses.index(min(losses))
        best_params = estimated_params[min_idx]
        estimator.init_vel = nn.Parameter(best_params['velocity'].to(estimator.device))

    return losses, estimated_params

def export_result(dataset, phys_args, estimator: Estimator, losses, estimated_params, config_id):
    save_attr = ['mpm_iter_cnt', 'rho', 'voxel_size', 'gravity', 'bc', 'fps', 'density_grid_size']
    pred = dict()
    pred['config_id'] = config_id
    v = estimator.init_vel.detach().cpu().numpy().tolist()
    pred['vel'] = v
    min_idx = losses.index(min(losses))
    best_params = estimated_params[min_idx]
    # best_params = estimated_params[-1]
    mat_params = dict()
    m = phys_args.material
    mat_params['material'] = m
    if (m == MPMSimulator.von_mises and estimator.simulator.non_newtonian == 1) or \
        m == MPMSimulator.viscous_fluid:
        # non_newtonian & newtonian
        mu = best_params['shear modulus']
        kappa = best_params['bulk modulus']
        mat_params['mu'] = mu
        mat_params['kappa'] = kappa
    else:
        # elasticity, drucker_prager, plasticine
        if 'Youngs modulus' in best_params and 'Poisson ratio' in best_params:
            E = best_params['Youngs modulus']
            nu = best_params['Poisson ratio']
        else:
            E = float((10 ** estimator.E).detach().cpu().numpy())
            nu = float((estimator.get_nu()).detach().cpu().numpy())
        mat_params['E'] = E
        mat_params['nu'] = nu

    if m == MPMSimulator.drucker_prager:
        mat_params['friction_alpha'] = best_params['friction angle']

    if m == MPMSimulator.von_mises:
        ys = best_params['Yield stress']
        mat_params['yield_stress'] = ys
        if estimator.simulator.non_newtonian == 1:
            eta = best_params['plastic viscosity']
            mat_params['plastic_viscosity'] = eta

    pred['mat_params'] = mat_params
    for attr in save_attr:
        pred[attr] = getattr(phys_args, attr)

    with open(os.path.join(dataset.model_path, f'{config_id}-pred.json'), 'w') as f:
        json.dump(pred, f, indent=4)

def get_args(parser: ArgumentParser, configpath):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline, unk = parser.parse_known_args(cmdlne_string)
    # config_file = args_cmdline.config_file
    model_path = None
    config_data = None
    gs_dict = None
    phys_args = None
    try:
        with open(configpath, 'r') as f:
            config_data = json.load(f)
            gs_dict = config_data.get("gs", None)
            phys_dict = config_data.get("physics", None)
            phys_args = Namespace(**phys_dict)
            model_path = gs_dict.get("model_path", None)

    except Exception as e:
        print('Parsing config file error')

    if model_path is None:
        model_path = getattr(args_cmdline, 'model_path', None)
    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    # except TypeError:
    except Exception as e:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    gs_merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            gs_merged_dict[k] = v

    if gs_dict is not None:
        for k, v in gs_dict.items():
            if v != None:
                gs_merged_dict[k] = v

    return Namespace(**gs_merged_dict), phys_args

def constraint_inv(y, bound):
    # return y
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale

if __name__ == "__main__":
    # Set up command line argument parser
    start_time = time.time()

    parser = ArgumentParser(description="Physical parameter estimation")
    #model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    seq_no = sys.argv[-1]
    print("Seq no ", seq_no )
    estimators = []
    phys_args_list = []
    gs_args_list = []
    datasets = []
    vols = []
    vol_densities_list = []
    cam_infos = []
    gts_list = []
    config_id = None
    optimizer_multi = None
    for i in range(10):
        configpath = f"data/MultiSeries/{seq_no}_{i}/config.json"
        model = ModelParams(parser)

        gs_args, phys_args = get_args(parser, configpath)
        config_id = phys_args.id
        print(phys_args)
        safe_state(gs_args.quiet)

        # 1. train def gs
        dataset = model.extract(gs_args)
        print(dataset.source_path)
        dataset.source_path = os.path.abspath(f"./data/MultiSeries/{seq_no}_{i}")
        dataset.config_path = os.path.abspath(f"./data/MultiSeries/{seq_no}_{i}/config.json")
        if not check_gs_model(dataset.model_path, gs_args.save_iterations):
            training(dataset, op.extract(gs_args), pipeline.extract(gs_args), gs_args.test_iterations + list(range(10000, 40001, 1000)), gs_args.save_iterations)
        torch.cuda.empty_cache()

        # 2. estimate velocity
        gts, vol, vol_densities, grid_size, volume_surface, cam_info = prepare_gt(model.extract(gs_args), gs_args.iteration, pipeline.extract(gs_args), phys_args)
        torch.cuda.empty_cache()
        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
        print('gt point count: {}'.format(gts[0].shape[0]))

        if i == 0:
            E = nn.Parameter(torch.tensor(getattr(phys_args, "init_E", 0.0), device="cuda"))
            yield_stress = nn.Parameter(torch.tensor([getattr(phys_args, "init_yield_stress", 0.0)], device="cuda"))
            plastic_viscosity = nn.Parameter(torch.tensor([getattr(phys_args, "init_plastic_viscosity", -1e6)], device="cuda"))
            friction_alpha = nn.Parameter(torch.tensor([getattr(phys_args, "init_friction_alpha", 0.0)], device="cuda"))
            global_mu = nn.Parameter(torch.tensor([np.log10(getattr(phys_args, "mu", 1.0))], device="cuda").float())
            global_kappa = nn.Parameter(torch.tensor([np.log10(getattr(phys_args, "kappa", 1.0))], device="cuda").float())

            nu_bound = getattr(phys_args, "nu_bound", [-0.99, 0.5])
            nu = nn.Parameter(constraint_inv(torch.tensor(getattr(phys_args, "init_nu", 0.0), device="cuda"), nu_bound))
            params = []
            for param_name, info in phys_args.params.items():
                if param_name == "Youngs modulus":
                    params.append({'params':E, 'lr':info.get('init_lr', 0.1), 'name': param_name})
                elif param_name == "Poisson ratio":
                    params.append({'params':nu, 'lr':info.get('init_lr', 0.025), 'name': param_name})
                elif param_name == "bulk modulus":
                    params.append({'params':global_kappa, 'lr':info.get('init_lr', 0.1), 'name': param_name})
                elif param_name == "shear modulus":
                    params.append({'params':global_mu, 'lr':info.get('init_lr', 0.1), 'name': param_name})
                elif param_name == "Yield stress":
                    params.append({'params':yield_stress, 'lr':info.get('init_lr', 0.1), 'name': param_name})
                elif param_name == "plastic viscosity":
                    params.append({'params':plastic_viscosity, 'lr':info.get('init_lr', 0.05), 'name': param_name})
                elif param_name == "friction angle":
                    params.append({'params':friction_alpha, 'lr':info.get('init_lr', 1.0), 'name': param_name})
            optimizer_multi = torch.optim.Adam([*params],amsgrad=False)

        estimator = Estimator(phys_args, 'float32', gts, surface_index=volume_surface, init_vol=vol, dynamic_scene=None,
                            image_scale=image_scale, pipeline=pipeline.extract(gs_args), image_op=op.extract(gs_args), optimizer_multi=optimizer_multi)
        estimator.set_stage(Estimator.velocity_stage)
        losses, e_s = train(estimator, phys_args, phys_args.vel_estimation_frames)
        torch.cuda.empty_cache()
        estimators.append(estimator)
        phys_args_list.append(phys_args)
        datasets.append(dataset)
        vols.append(vol)
        vol_densities_list.append(vol_densities)
        cam_infos.append(cam_info)
        gts_list.append(gts)
        # 3. estimate physical parameters
        # scene = train_gs_with_fixed_pcd(vol, dataset, op.extract(gs_args),
        #                                 pipeline.extract(gs_args),
        #                                 gs_args.test_iterations + list(range(10000, 40001, 1000)),
        #                                 gs_args.save_iterations, None, phys_args.fps, True,
        #                                 cam_info, phys_args.density_grid_size)
    for i in range(10):
        scene = assign_gs_to_pcd(vols[i], vol_densities_list[i], datasets[i], op.extract(gs_args_list[i]),
                                        pipeline.extract(gs_args_list[i]),
                                        cam_infos[i], phys_args_list[i].density_grid_size)
        estimators[i].set_scene(scene)

        estimators[i].set_stage(Estimator.physical_params_stage)
    max_f = len(gts_list[0])
    losses, e_s = train_multi(estimators, phys_args_list, max_f)
    print(phys_args)
    print(estimator.init_vel)
    print(config_id)
    export_result(datasets[0], phys_args_list[0], estimators[0], losses, e_s, config_id)
    print("consume time {}".format(time.time() - start_time))
