import numpy as np
import sys
import ufl

import petsc4py
petsc4py.init(sys.argv)
import slepc4py
slepc4py.init(sys.argv)

from dolfinx.io import XDMFFile
import dolfinx.fem.petsc
from dolfinx import fem
from mpi4py import MPI
from slepc4py import SLEPc

import meshio

def create_operators(domain:dolfinx.mesh.Mesh,
                     cell_tags:dolfinx.mesh.MeshTags,
                     cell_data:dict,
                     material_field:str,
                     fiber_field:str) -> tuple[petsc4py.PETSc.Mat, petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]:

    """
    Create PETSc operators for eigenproblem with FEM using lumped mass matrix

    in: 
        domain (dolfinx.mesh.Mesh) -> Mesh domain
        cell_tags (dolfinx.mesh.MeshTags) -> Cell numeration (dolfinx)
        cell_data (dict) -> Cell data dict (meshio)
        material_field (str) -> Name of organ field in file ("OrganID")
        fiber_field (str) -> Name of fiber field in file ("Fibers")
    out:
        A (PETSc.Mat) -> Stiffness matrix
        A_tilde (PETSc.Mat) -> Diagonally scaled stiffness matrix
        d (PETSc.Vec) -> Scaling vector
    """

    # discontinuous Galerkin function space
    DG0 = fem.functionspace(domain, ("DG",0)) 
    
    # initialize fiber direction function
    Fiber = [fem.Function(DG0), fem.Function(DG0), fem.Function(DG0)] 
    
    # intra/extra cellular conductivity
    Sigma_i = [fem.Function(DG0), fem.Function(DG0), fem.Function(DG0)] 
    Sigma_e = [fem.Function(DG0), fem.Function(DG0), fem.Function(DG0)]
    
    # dimension
    Dim = len(Sigma_i) 
    
    # get material tags in dolfinx order
    materialField = cell_data[material_field][0][cell_tags.values] 

    # set fiber directions
    for i in range(Dim):
        with Fiber[i].vector.localForm() as loc_fiber:
            loc_fiber.setValues(cell_tags.indices, cell_data[fiber_field][0][cell_tags.values, i])

    # set of (unique) material tags
    M = list(np.unique(cell_data[material_field][0]))

    conducting_regions = {
        "SAN": [[1],[0.0008, 0.0008*1.0, 0.0008*1.0],[0.0008, 0.0008*1.0, 0.0008*1.0]],
        "CT": [[2],[0.0085, 0.0085*0.15, 0.0085*0.15],[0.0085, 0.0085*0.15, 0.0085*0.15]],
        "BBL": [[5, 7, 8, 9, 10],[0.0075, 0.0075*0.15, 0.0075*0.15],[0.0075, 0.0075*0.15, 0.0075*0.15]],
        "BBR": [[3, 4],[0.0075, 0.0075*0.15, 0.0075*0.15],[0.0075, 0.0075*0.15, 0.0075*0.15]],
        "BB (ins)": [[6, 7],[0.0, 0.0*1.0, 0.0*1.0],[0.0, 0.0*1.0, 0.0*1.0]],
        "IB": [[11],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "RAS": [[12],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "RLW": [[13],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "RAA": [[14],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "PM": [[15, 16, 17, 18, 19, 20, 21, 22, 23, 24],[0.0075, 0.0075*0.15, 0.0075*0.15],[0.0075, 0.0075*0.15, 0.0075*0.15]],
        "IST": [[25],[0.0015, 0.0015*1.0, 0.0015*1.0],[0.0015, 0.0015*1.0, 0.0015*1.0]],
        "SCV": [[26],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "ICV": [[27],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "TV": [[28],[0.0030, 0.0030*0.35, 0.0030*0.35],[0.0030, 0.0030*0.35, 0.0030*0.35]],
        "LFO": [[29],[0.0075, 0.0075*0.15, 0.0075*0.15],[0.0075, 0.0075*0.15, 0.0075*0.15]],
        "FO (ins)": [[30],[0.0, 0.0*1.0, 0.0*1.0],[0.0, 0.0*1.0, 0.0*1.0]],
        "LSW": [[31, 32],[0.0030, 0.0030*0.25, 0.0030*0.25],[0.0030, 0.0030*0.25, 0.0030*0.25]],
        "LAS": [[33],[0.0030, 0.0030*0.25, 0.0030*0.25],[0.0030, 0.0030*0.25, 0.0030*0.25]],
        "LAA": [[34],[0.0030, 0.0030*0.25, 0.0030*0.25],[0.0030, 0.0030*0.25, 0.0030*0.25]],
        "LPW": [[35],[0.0030, 0.0030*0.25, 0.0030*0.25],[0.0030, 0.0030*0.25, 0.0030*0.25]],
        "MV": [[36, 37],[0.0030, 0.0030*0.25, 0.0030*0.25],[0.0030, 0.0030*0.25, 0.0030*0.25]],
        "RPV": [[38, 39, 40, 41, 42, 43],[0.0017, 0.0017*0.5, 0.0017*0.5],[0.0017, 0.0017*0.5, 0.0017*0.5]],
        "LPV": [[44, 45, 46, 47, 48, 49],[0.0017, 0.0017*0.5, 0.0017*0.5],[0.0017, 0.0017*0.5, 0.0017*0.5]],
        "CS": [[50, 51, 52, 53],[0.0060, 0.0060*0.5, 0.0060*0.5],[0.0060, 0.0060*0.5, 0.0060*0.5]],
        }
    
    # assign conductivity according to each marker
    for marker in M:
        marked_cells = cell_tags.indices[materialField == marker]
        
        for i in range(Dim):
            for _, region_data in conducting_regions.items(): 
                if marker in region_data[0]:
                    with Sigma_i[i].vector.localForm() as loc_i:
                        loc_i.setValues(marked_cells, np.full(len(marked_cells), region_data[1][i]))
                    with Sigma_e[i].vector.localForm() as loc_e:
                        loc_e.setValues(marked_cells, np.full(len(marked_cells), region_data[2][i]))
                    break

    # assert tensor positivity
    assert np.all(Sigma_i[0].x.array > 0) and np.all(Sigma_i[1].x.array > 0) and np.all(Sigma_i[2].x.array > 0)
    assert np.all(Sigma_e[0].x.array > 0) and np.all(Sigma_e[1].x.array > 0) and np.all(Sigma_e[2].x.array > 0)

    # set Fiber as ufl vector
    FiberField = ufl.as_vector(Fiber)

    # create intra/extra-cellular conductivity tensors
    G_i = Sigma_i[0]*ufl.outer(FiberField, FiberField) + Sigma_i[1]*(ufl.Identity(3) - ufl.outer(FiberField, FiberField))
    G_e = Sigma_e[0]*ufl.outer(FiberField, FiberField) + Sigma_e[1]*(ufl.Identity(3) - ufl.outer(FiberField, FiberField))

    # compute bulk conductivity tensor
    G = G_i*ufl.inv(G_i+G_e)*G_e   

    # FEM function space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # form corresponding to lhs
    a = ufl.dot(G*ufl.grad(u),ufl.grad(v))*ufl.dx
    
    # form corresponding to rhs
    b = u*v*ufl.dx
    
    # assemble stifness
    A = fem.petsc.assemble_matrix(fem.form(a))
    A.assemble()

    # create and assemble lumped mass matrix
    one = fem.Function(V)
    one.vector.array_w[:] = 1.0
    B_lump = fem.petsc.assemble_vector(fem.form(ufl.action(b,one)))
    B_lump.assemble()

    d = B_lump.copy()
    d.array_w[:] = 1/np.sqrt(d.array_r)
    
    A_tilde = A.copy()
    A_tilde.diagonalScale(d,d)

    return A, A_tilde, d

def solve_ep(A:petsc4py.PETSc.Mat,
             A_tilde:petsc4py.PETSc.Mat,
             d:petsc4py.PETSc.Vec,
             num:int) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Solve eigenproblem with SLEPc

    in: 
        A (PETSc.Mat) -> Stiffness matrix
        A_tilde (PETSc.Mat) -> Diagonally scaled stiffness matrix
        d (PETSc.Mat) -> Scaling vector
        num (int) -> Number of eigenvalues to compute
    out:
        eig_vecs (np.ndarray) -> Eigenvector array
        eig_vals (np.ndarray) -> Eigenvalue array
    """
    
    print(f"solving eigenproblem ...")

    # initialize solver
    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setFromOptions()
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
    
    shift = SLEPc.ST().create(MPI.COMM_WORLD)
    shift.setType(SLEPc.ST.Type.SINVERT)
    shift.setShift(sys.float_info.epsilon)
    eigensolver.setST(shift)
    
    # compute eigenvalues in increasing magnitude
    eigensolver.setWhichEigenpairs(eigensolver.Which.TARGET_MAGNITUDE)
    eigensolver.setDimensions(nev = num)
    
    # set operator and solve
    eigensolver.setOperators(A_tilde)
    eigensolver.solve()
    
    evs = eigensolver.getConverged()

    print(f"# converged eigenpairs: {evs}")
    
    # extract eigenvalues and eigenvectors
    eig_vecs = []
    eig_vals = []
    
    for i in range(num):
        vr, vi = A.getVecs()
        l = eigensolver.getEigenpair(i, vr, vi)
        vr.array_w[:] = d.array_r * vr.array_r
        eig_vecs.append(vr.getArray().copy())
        eig_vals.append(l.real)
    
    eig_vecs = np.array(eig_vecs).T
    eig_vals= np.array(eig_vals)
    
    return eig_vecs, eig_vals

def solve_ep_model(model_path:str = "xdmf/AtriaVoxHexa.xdmf",
                   ct_path:str = "xdmf/cellTags.xdmf",
                   out_path:str = None,
                   num:int = 100,
                   material_field:str = "OrganID",
                   fiber_field:str = "Fibers") -> tuple[np.ndarray, np.ndarray]:
    
    """
    Solve eigenproblem with specified file paths for geometry

    in: 
        model_path (str) -> Path for .xdmf geometry with corresponding properties (read via meshio)
        ct_path (str) -> Path for .xdmf meshtags file (read via XDMFFile)
        out_path (str) -> Path for .npz output 
        num (int) -> Number of eigenpairs to consider
        material_field (str) -> Name of organ field in file ("OrganID")
        fiber_field (str) -> Name of fiber field in file ("Fibers")
    out:
        eig_vecs (np.ndarray)
        eig_vals (np.ndarray)
    """ 
    
    # read mesh and cell tags from .xdmf file
    with XDMFFile(MPI.COMM_WORLD, ct_path, "r") as xdmf:
        domain = xdmf.read_mesh(name = "Grid")
        cell_tags = xdmf.read_meshtags(domain, name = "Grid")
    
    # extract cell data
    cell_data = meshio.xdmf.read(model_path).cell_data

    # normalize geometry
    verts = domain.geometry.x
    centroid = verts.mean(0)
    std_max = verts.std(0).max()
    verts_new = (verts - centroid)/std_max
    
    domain.geometry.x[:] = verts_new

    # create operators for eigenproblem and solve
    A, A_tilde, d = create_operators(domain = domain, cell_tags = cell_tags, cell_data = cell_data, material_field = material_field, fiber_field = fiber_field)
    eig_vecs, eig_vals = solve_ep(A = A, A_tilde = A_tilde, d = d, num = num)

    # save output to .npz file
    if out_path is not None:
        np.savez(out_path, geometry = domain.geometry.x, eig_vecs = eig_vecs, eig_vals = eig_vals)

    return eig_vecs, eig_vals

def kernel_signature(eig_vecs:np.ndarray,
                     eig_vals:np.ndarray,
                     t_steps:int,
                     times:np.ndarray = None,
                     num:int = 100,
                     scale:np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Penk, D. (2024). mesh-signatures. GitHub. https://github.com/DominikPenk/mesh-signatures

    Computes kernel signature from eigenpairs for a given number of timesteps

    in:
        eig_vecs (np.ndarray) -> Eigenvector array 
        eig_vals (np.ndarray) -> Eigenvalue array
        t_steps (int) -> Number of timesteps for signature evolution
        times (np.array) -> Array of times to consider. Length must match t_steps. If None, times are computed from eigenvalues
        num (int) -> Number of eigenvalues to consider for signature computation
        scale (np.array) -> Scaling factor for each time. If None, trace is used
    out:
        K (np.ndarray) -> kernel signature array
        T (np.ndarray) -> times array
    """  
    eig_n = min(num, len(eig_vals))
    
    assert eig_n <= len(eig_vecs[0,:]), "number of used eigenvalues must be at most number of eigenvectors available"
    
    # set time array with log progression
    if times is None: 

        t_min  = 4 * np.log(10) / eig_vals[eig_n - 1]
        t_max  = 4 * np.log(10) / eig_vals[1]

        T = np.geomspace(t_min, t_max, t_steps)
    
    else: 

        T = np.array(times).flatten()
        assert len(T) == t_steps # assert dimensionality
        
    # compute signature
    phi2 = np.square(eig_vecs[:, 1:eig_n]) # {phi_i(x)*2}_{i in [1,...,eig_num]}
    exp_ = np.exp(-eig_vals[1:eig_n, None]*T[None]) # {exp(-lambda_i)t}_{i in [eig_n], t in T}
    K = np.sum(phi2[..., None]*exp_[None], axis=1) # {k_t(x,x)}_{t in T}
    
    # apply scaling
    if scale is None:
        trace = np.sum(exp_, axis=0)
    else:
        trace = scale
    
    K = K/trace[None]
    
    return K, T

def solve_fk_model(model_path:str = "xdmf/AtriaVoxHexa.xdmf",
                   ct_path:str = "xdmf/cellTags.xdmf",
                   out_path:str = None,
                   num:int = 100,
                   material_field:str = "OrganID",
                   fiber_field:str = "Fibers") -> None:
    
    """
    Compute fks in geometry with specified file paths (saves .xdmf)

    in: 
        model_path (str) -> Path for .xdmf geometry with corresponding properties (read via meshio)
        ct_path (str) -> Path for .xdmf meshtags file (read via XDMFFile)
        out_path (str) -> Path for .npz output 
        num (int) -> Number of eigenpairs to consider
        material_field (str) -> Name of organ field in file ("OrganID")
        fiber_field (str) -> Name of fiber field in file ("Fibers")
    out:
        None
    """ 
    
    with XDMFFile(MPI.COMM_WORLD, ct_path, "r") as xdmf:
        domain = xdmf.read_mesh(name = "Grid")
        cell_tags = xdmf.read_meshtags(domain, name = "Grid")
    
    cell_data = meshio.xdmf.read(model_path).cell_data

    verts = domain.geometry.x
    centroid = verts.mean(0)
    std_max = verts.std(0).max()
    verts_new = (verts - centroid)/std_max
    
    domain.geometry.x[:] = verts_new

    A, A_tilde, d = create_operators(domain = domain, cell_tags = cell_tags, cell_data = cell_data, material_field = material_field, fiber_field = fiber_field)
    eig_vecs, eig_vals = solve_ep(A = A, A_tilde = A_tilde, d = d, num = num)

    K, T = kernel_signature(eig_vecs = eig_vecs, eig_vals = eig_vals, t_steps = 50)

    if out_path is not None:
        np.savez(out_path, geometry = domain.geometry.x, fks = K, times = T)
    
    V = fem.functionspace(mesh = domain, element = ("Lagrange", 1))
    k = fem.Function(V)
    
    with XDMFFile(MPI.COMM_WORLD, f"fks.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        
        for i in range(len(T)):
            k.x.array[:] = K[:,i]
            xdmf.write_function(k, T[i])
    
    return None

solve_fk_model()
