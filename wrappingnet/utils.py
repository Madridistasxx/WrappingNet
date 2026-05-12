##################################################
#
#   Copyright (c) 2010-2024, InterDigital
#   All rights reserved.
#
#   See LICENSE under the root folder.
#
##################################################

import numpy as np
import torch
from torch_geometric.data import Data


def get_base_mesh(pos, faces, n_iter=3):
    # assumes faces
    for _ in range(n_iter):
        nf = faces.shape[0] // 4  # output number of faces
        faces = torch.stack(
            (faces[0:nf, 0], faces[nf : 2 * nf, 0], faces[2 * nf : 3 * nf, 0]), dim=1
        )

        num_nodes = len(torch.unique(faces.flatten()))
        pos = pos[0:num_nodes]
    return pos, faces


def compute_face_adjacency(faces):
    """
    Fast boundary-safe face adjacency.

    faces: [F, 3]

    return:
        FAF: [F, 3]
        FAF[i, j] = neighboring face across local edge j.
        Boundary edge: FAF[i, j] = i.
        Non-manifold edge: choose one other adjacent face.
    """
    faces = faces.long().contiguous()
    device = faces.device
    F = faces.shape[0]

    # Keep the same local edge order as the current dev version:
    # local 0: v0-v1
    # local 1: v1-v2
    # local 2: v2-v0
    edges = torch.cat(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        dim=0,
    )
    edges = torch.sort(edges, dim=1)[0]

    face_ids = torch.arange(F, device=device).repeat(3)

    local_ids = torch.cat(
        (
            torch.zeros(F, dtype=torch.long, device=device),
            torch.ones(F, dtype=torch.long, device=device),
            2 * torch.ones(F, dtype=torch.long, device=device),
        ),
        dim=0,
    )

    unique_edges, inverse, counts = torch.unique(
        edges,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    # Default: boundary edge points to self.
    FAF = torch.arange(F, device=device).view(F, 1).repeat(1, 3)

    # Sort half-edges by edge id.
    order = torch.argsort(inverse)
    face_sorted = face_ids[order]
    local_sorted = local_ids[order]

    starts = torch.cumsum(counts, dim=0) - counts

    # Fast path: normal manifold edges, count == 2.
    mask2 = counts == 2
    starts2 = starts[mask2]

    if starts2.numel() > 0:
        p0 = starts2
        p1 = starts2 + 1

        f0 = face_sorted[p0]
        f1 = face_sorted[p1]
        l0 = local_sorted[p0]
        l1 = local_sorted[p1]

        FAF[f0, l0] = f1
        FAF[f1, l1] = f0

    # Rare path: non-manifold edges, count > 2.
    # Usually few or zero. Python loop here is acceptable.
    mask_gt2 = counts > 2
    starts_gt2 = starts[mask_gt2]
    counts_gt2 = counts[mask_gt2]

    for s, c in zip(starts_gt2.tolist(), counts_gt2.tolist()):
        fs = face_sorted[s : s + c]
        ls = local_sorted[s : s + c]

        for k in range(c):
            f = fs[k]
            le = ls[k]
            others = fs[fs != f]

            if others.numel() > 0:
                FAF[f, le] = others[0]
            else:
                FAF[f, le] = f

    return FAF


def extract_features(pos, faces):
    # order invariant features
    vecs1 = pos[faces[:, 1]] - pos[faces[:, 0]]
    vecs2 = pos[faces[:, 2]] - pos[faces[:, 1]]
    vecs3 = pos[faces[:, 0]] - pos[faces[:, 2]]
    a = torch.linalg.norm(vecs1, dim=1)
    b = torch.linalg.norm(vecs2, dim=1)
    c = torch.linalg.norm(vecs3, dim=1)
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    face_normals = torch.cross(vecs1, vecs2, dim=1)
    center_pos = (pos[faces[:, 0]] + pos[faces[:, 1]] + pos[faces[:, 2]]) / 3
    FAF = compute_face_adjacency(faces)
    pos0, pos1, pos2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    center_neigh0 = (pos0[FAF[:, 0]] + pos1[FAF[:, 0]] + pos2[FAF[:, 0]]) / 3
    center_neigh1 = (pos0[FAF[:, 1]] + pos1[FAF[:, 1]] + pos2[FAF[:, 1]]) / 3
    center_neigh2 = (pos0[FAF[:, 2]] + pos1[FAF[:, 2]] + pos2[FAF[:, 2]]) / 3
    center_neigh = (center_neigh0 + center_neigh1 + center_neigh2) / 3
    curve_vec = center_pos - center_neigh
    # bary = get_barycenters(pos, faces)
    feats = torch.cat(
        (
            area_sq.unsqueeze(1),
            face_normals,
            curve_vec,
            # bary
        ),
        dim=1,
    )
    return feats


def extract_features_local(pos, faces, FAF=None):
    # order invariant features
    vecs1 = pos[faces[:, 1]] - pos[faces[:, 0]]
    vecs2 = pos[faces[:, 2]] - pos[faces[:, 1]]
    vecs3 = pos[faces[:, 0]] - pos[faces[:, 2]]
    a = torch.linalg.norm(vecs1, dim=1)
    b = torch.linalg.norm(vecs2, dim=1)
    c = torch.linalg.norm(vecs3, dim=1)
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    face_normals = torch.cross(vecs1, vecs2, dim=1)
    center_pos = (pos[faces[:, 0]] + pos[faces[:, 1]] + pos[faces[:, 2]]) / 3
    if FAF is None:
        FAF = compute_face_adjacency(faces)
    pos0, pos1, pos2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    center_neigh0 = (pos0[FAF[:, 0]] + pos1[FAF[:, 0]] + pos2[FAF[:, 0]]) / 3
    center_neigh1 = (pos0[FAF[:, 1]] + pos1[FAF[:, 1]] + pos2[FAF[:, 1]]) / 3
    center_neigh2 = (pos0[FAF[:, 2]] + pos1[FAF[:, 2]] + pos2[FAF[:, 2]]) / 3
    center_neigh = (center_neigh0 + center_neigh1 + center_neigh2) / 3
    curve_vec = center_pos - center_neigh
    # bary = get_barycenters(pos, faces)
    feats = torch.cat(
        (
            area_sq.unsqueeze(1),
            face_normals,
            curve_vec,
            # bary
        ),
        dim=1,
    )
    return feats


def normalize_pos(pos, out_norm=10):
    # centralize
    centroid = torch.mean(pos, dim=0)
    pos = pos - centroid
    # normalize in ball of radius 10
    m = torch.max(torch.sqrt(torch.sum(pos**2, dim=1)))
    pos = out_norm * pos / m  # scaling
    return pos


def gen_sphere_samples(n):
    indices = torch.arange(n) + 0.5

    phi = torch.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = (
        torch.cos(theta) * torch.sin(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(phi),
    )
    return torch.stack((x, y, z), dim=1)


def get_barycenters(pos, face_list):
    """
    Given list of faces, return barycenters of vertex positions.
    pos: [num_nodes, 3]
    face_list: [num_faces, 3]
    """
    tri_pos = torch.stack(
        (pos[face_list[:, 0]], pos[face_list[:, 1]], pos[face_list[:, 2]]), dim=2
    )  # [num_faces, 3, 3]
    bary = torch.mean(tri_pos, dim=2)  # [num_faces, 3]
    return bary


def loop_subdivide_once_like_wrappingnet(pos, faces):
    """
    Replicates WrappingNet LoopUnPool.loop_subdivision() connectivity/order.

    pos:   [N, 3]
    faces: [F, 3]

    returns:
        pos_new:     [N_new, 3]
        faces_new:   [4F, 3]
        face_parent: [4F]
    """
    pos = pos.contiguous()
    faces = faces.long().contiguous()

    num_nodes = pos.shape[0]
    num_faces = faces.shape[0]
    device = pos.device

    # Half-edges: edge opposite local vertex naming follows WrappingNet code.
    hE = torch.cat(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        dim=0,
    )

    # Undirected unique edges.
    hE = torch.sort(hE, dim=1)[0]
    E, hE2E = torch.unique(hE, dim=0, return_inverse=True)

    # New vertices are edge midpoints.
    newV = (pos[E[:, 0], :] + pos[E[:, 1], :]) / 2.0
    pos_new = torch.cat((pos, newV), dim=0)

    # These are temporary half-edge vertex ids before mapping shared edges.
    E2 = num_nodes + torch.arange(num_faces, device=device)
    E0 = num_nodes + num_faces + torch.arange(num_faces, device=device)
    E1 = num_nodes + 2 * num_faces + torch.arange(num_faces, device=device)

    # Important: this is the same 4-block face order as WrappingNet LoopUnPool.
    faces_tmp = torch.cat(
        (
            torch.stack((faces[:, 0], E2, E1), dim=1),
            torch.stack((faces[:, 1], E0, E2), dim=1),
            torch.stack((faces[:, 2], E1, E0), dim=1),
            torch.stack((E0, E1, E2), dim=1),
        ),
        dim=0,
    )

    # Map temporary half-edge ids to unique edge midpoint ids.
    hE2E_full = torch.cat(
        (
            torch.arange(num_nodes, device=device),
            hE2E + num_nodes,
        ),
        dim=0,
    )

    faces_new = hE2E_full[faces_tmp]

    # Each of the 4 child blocks comes from the same parent face.
    face_parent = torch.cat(
        [torch.arange(num_faces, device=device)] * 4,
        dim=0,
    )

    assert faces_new.min() >= 0
    assert faces_new.max() < pos_new.shape[0], (
        faces_new.max().item(),
        pos_new.shape[0],
    )

    return pos_new, faces_new, face_parent


def build_ddacs_hierarchy(pos_l0, face_l0, num_levels=3):
    """
    DDACS original triangle mesh = level 0.
    Generate level 1/2/3 via WrappingNet-compatible subdivision.

    pos_l0:  [N0, 3]
    face_l0: [F0, 3]

    returns Data with:
        pos_l0, face_l0
        pos_l1, face_l1
        pos_l2, face_l2
        pos_l3, face_l3

    Convention:
        face_l* stored as [3, F] inside PyG Data.
    """
    pos_list = [pos_l0.float().contiguous()]
    face_list = [face_l0.long().contiguous()]
    parent_list = []

    pos = pos_list[0]
    faces = face_list[0]

    for _ in range(num_levels):
        pos, faces, parent = loop_subdivide_once_like_wrappingnet(pos, faces)
        pos_list.append(pos)
        face_list.append(faces)
        parent_list.append(parent)

    FAF_list = [compute_face_adjacency(face_list[i]) for i in range(len(face_list))]

    data = Data(
        pos=pos_list[-1],
        face=face_list[-1].T.contiguous(),
        pos_l0=pos_list[0],
        face_l0=face_list[0].T.contiguous(),
        FAF_l0=FAF_list[0],
        pos_l1=pos_list[1],
        face_l1=face_list[1].T.contiguous(),
        FAF_l1=FAF_list[1],
        pos_l2=pos_list[2],
        face_l2=face_list[2].T.contiguous(),
        FAF_l2=FAF_list[2],
        pos_l3=pos_list[3],
        face_l3=face_list[3].T.contiguous(),
        FAF_l3=FAF_list[3],
        face_parent_1_to_0=parent_list[0],
        face_parent_2_to_1=parent_list[1],
        face_parent_3_to_2=parent_list[2],
    )

    return data


def ddacs_quad_to_tri_data(tool_batch, num_levels=3):
    """
    Convert DDACS tool batch into WrappingNet hierarchy.

    Assumes:
        tool_batch.x    = vertices, usually [N, 3] or [1, N, 3]
        tool_batch.face = quad faces as [4, F] or [F, 4]

    returns:
        PyG Data where data.pos/data.face are level-3 target.
    """
    pos = tool_batch.x

    # Handle possible batch dimension. Keep batch_size=1 for now.
    if pos.dim() == 3:
        if pos.shape[0] != 1:
            raise ValueError(
                f"Expected batch_size=1 for WrappingNet, got pos shape {pos.shape}"
            )
        pos = pos[0]

    pos = pos[:, :3].float().contiguous()

    face = tool_batch.face.long()

    if face.dim() != 2:
        raise ValueError(f"Expected 2D face tensor, got {face.shape}")

    # Convert quad face to [F, 4].
    if face.shape[0] == 4:
        quads = face.T.contiguous()
    elif face.shape[1] == 4:
        quads = face.contiguous()
    elif face.shape[0] == 3:
        # Already triangle [3, F].
        tris = face.T.contiguous()
        assert tris.max() < pos.shape[0], (tris.max().item(), pos.shape[0])
        return build_ddacs_hierarchy(pos, tris, num_levels=num_levels)
    elif face.shape[1] == 3:
        # Already triangle [F, 3].
        tris = face.contiguous()
        assert tris.max() < pos.shape[0], (tris.max().item(), pos.shape[0])
        return build_ddacs_hierarchy(pos, tris, num_levels=num_levels)
    else:
        raise ValueError(f"Unknown face shape: {face.shape}")

    # Split each quad [a,b,c,d] into two triangles.
    tri1 = quads[:, [0, 1, 2]]
    tri2 = quads[:, [0, 2, 3]]
    tris = torch.cat([tri1, tri2], dim=0).contiguous()  # [F_tri, 3]

    assert tris.min() >= 0
    assert tris.max() < pos.shape[0], (tris.max().item(), pos.shape[0])

    return build_ddacs_hierarchy(pos, tris, num_levels=num_levels)
