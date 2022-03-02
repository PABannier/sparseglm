use pyo3::prelude::*;

mod block_mcp;
mod lasso;
mod mcp;
mod mtl_lasso;

#[pymodule]
fn sparseglm_solver(_py: Python, m: &PyModule) -> PyResult<()> {
    // Lasso
    m.add_class::<lasso::LassoWrapper>()?;
    m.add_class::<mtl_lasso::MultiTaskLassoWrapper>()?;
    // MCP
    m.add_class::<mcp::MCPWrapper>()?;
    m.add_class::<block_mcp::BlockMCPWrapper>()?;
    Ok(())
}
