// 810s
module attributes { "simulation.dims" = "ARRAY_0:257x257x257;ARRAY_1:257x257x257;ARRAY_2:257x257;ARRAY_3:257;ARRAY_4:257;ARRAY_5:257x257x257;ARRAY_6:257x257;ARRAY_7:257;ARRAY_8:257;ARRAY_9:257x257x257;ARRAY_10:257;ARRAY_11:257;ARRAY_12:257x257;ARRAY_13:257x257" } {
  func.func @kernel_fdtd_apml(
      %mui: f32, %ch: f32,
      %Ax: memref<?x?xf32>, %Ry: memref<?x?xf32>,
      %clf: memref<?x?xf32>, %tmp: memref<?x?xf32>,
      %Bza: memref<?x?x?xf32>, %Ex: memref<?x?x?xf32>,
      %Ey: memref<?x?x?xf32>, %Hz: memref<?x?x?xf32>,
      %czm: memref<?xf32>, %czp: memref<?xf32>,
      %cxmh: memref<?xf32>, %cxph: memref<?xf32>,
      %cymh: memref<?xf32>, %cyph: memref<?xf32>
  ) {
    affine.for %loop_once = 0 to 1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Main computation loops
    affine.for %iz = 0 to 256 {
      affine.for %iy = 0 to 256 {
        // Body 1: Loop over inner x-dimension (ix from 0 to cxm-1)
        affine.for %ix = 0 to 256 {
          // clf[iz,iy] = Ex[iz,iy,ix] - Ex[iz,iy+1,ix] + Ey[iz,iy,ix+1] - Ey[iz,iy,ix]
          %ex_curr = affine.load %Ex[%iz, %iy, %ix] : memref<?x?x?xf32>
          %ex_next = affine.load %Ex[%iz, %iy + 1, %ix] : memref<?x?x?xf32>
          %ey_next = affine.load %Ey[%iz, %iy, %ix + 1] : memref<?x?x?xf32>
          %ey_curr = affine.load %Ey[%iz, %iy, %ix] : memref<?x?x?xf32>
          
          %sub_ex = arith.subf %ex_curr, %ex_next : f32
          %sub_ey = arith.subf %ey_next, %ey_curr : f32
          %clf_val = arith.addf %sub_ex, %sub_ey : f32
          affine.store %clf_val, %clf[%iz, %iy] : memref<?x?xf32>

          // tmp[iz,iy] = (cymh[iy]/cyph[iy])*Bza[iz,iy,ix] - (ch/cyph[iy])*clf[iz,iy]
          %cymh_val = affine.load %cymh[%iy] : memref<?xf32>
          %cyph_val = affine.load %cyph[%iy] : memref<?xf32>
          %bza_val = affine.load %Bza[%iz, %iy, %ix] : memref<?x?x?xf32>
          
          %div_cymh = arith.divf %cymh_val, %cyph_val : f32
          %term1 = arith.mulf %div_cymh, %bza_val : f32
          %div_ch = arith.divf %ch, %cyph_val : f32
          %term2 = arith.mulf %div_ch, %clf_val : f32
          %tmp_val = arith.subf %term1, %term2 : f32
          affine.store %tmp_val, %tmp[%iz, %iy] : memref<?x?xf32>

          // Hz[iz,iy,ix] = (cxmh[ix]/cxph[ix])*Hz[iz,iy,ix] 
          //               + (mui*czp[iz]/cxph[ix])*tmp[iz,iy]
          //               - (mui*czm[iz]/cxph[ix])*Bza[iz,iy,ix]
          %cxmh_val = affine.load %cxmh[%ix] : memref<?xf32>
          %cxph_val = affine.load %cxph[%ix] : memref<?xf32>
          %hz_old = affine.load %Hz[%iz, %iy, %ix] : memref<?x?x?xf32>
          %czp_val = affine.load %czp[%iz] : memref<?xf32>
          %czm_val = affine.load %czm[%iz] : memref<?xf32>
          
          %div_cxmh = arith.divf %cxmh_val, %cxph_val : f32
          %hz_term1 = arith.mulf %div_cxmh, %hz_old : f32
          %mui_czp = arith.mulf %mui, %czp_val : f32
          %div_czp = arith.divf %mui_czp, %cxph_val : f32
          %hz_term2 = arith.mulf %div_czp, %tmp_val : f32
          %mui_czm = arith.mulf %mui, %czm_val : f32
          %div_czm = arith.divf %mui_czm, %cxph_val : f32
          %hz_term3 = arith.mulf %div_czm, %bza_val : f32
          
          %sum_hz = arith.addf %hz_term1, %hz_term2 : f32
          %hz_new = arith.subf %sum_hz, %hz_term3 : f32
          affine.store %hz_new, %Hz[%iz, %iy, %ix] : memref<?x?x?xf32>
          
          // Bza[iz,iy,ix] = tmp[iz,iy]
          affine.store %tmp_val, %Bza[%iz, %iy, %ix] : memref<?x?x?xf32>
        }

        // Body 2: Handle last x-element (ix = cxm)
        // clf[iz,iy] = Ex[iz,iy,cxm] - Ex[iz,iy+1,cxm] + Ry[iz,iy] - Ey[iz,iy,cxm]
        %ex_curr_b2 = affine.load %Ex[%iz, %iy, 256] : memref<?x?x?xf32>
        %ex_next_b2 = affine.load %Ex[%iz, %iy + 1, 256] : memref<?x?x?xf32>
        %ry_val = affine.load %Ry[%iz, %iy] : memref<?x?xf32>
        %ey_curr_b2 = affine.load %Ey[%iz, %iy, 256] : memref<?x?x?xf32>
        
        %sub_ex_b2 = arith.subf %ex_curr_b2, %ex_next_b2 : f32
        %add_ry = arith.addf %sub_ex_b2, %ry_val : f32
        %clf_val_b2 = arith.subf %add_ry, %ey_curr_b2 : f32
        affine.store %clf_val_b2, %clf[%iz, %iy] : memref<?x?xf32>
        
        // tmp[iz,iy] = (cymh[iy]/cyph[iy])*Bza[iz,iy,cxm] - (ch/cyph[iy])*clf[iz,iy]
        %cymh_val_b2 = affine.load %cymh[%iy] : memref<?xf32>
        %cyph_val_b2 = affine.load %cyph[%iy] : memref<?xf32>
        %bza_val_b2 = affine.load %Bza[%iz, %iy, 256] : memref<?x?x?xf32>
        
        %div_cymh_b2 = arith.divf %cymh_val_b2, %cyph_val_b2 : f32
        %term1_b2 = arith.mulf %div_cymh_b2, %bza_val_b2 : f32
        %div_ch_b2 = arith.divf %ch, %cyph_val_b2 : f32
        %term2_b2 = arith.mulf %div_ch_b2, %clf_val_b2 : f32
        %tmp_val_b2 = arith.subf %term1_b2, %term2_b2 : f32
        affine.store %tmp_val_b2, %tmp[%iz, %iy] : memref<?x?xf32>
        
        // Hz[iz,iy,cxm] = (cxmh[cxm]/cxph[cxm])*Hz[iz,iy,cxm] 
        //               + (mui*czp[iz]/cxph[cxm])*tmp[iz,iy]
        //               - (mui*czm[iz]/cxph[cxm])*Bza[iz,iy,cxm]
        %cxmh_val_b2 = affine.load %cxmh[256] : memref<?xf32>
        %cxph_val_b2 = affine.load %cxph[256] : memref<?xf32>
        %hz_old_b2 = affine.load %Hz[%iz, %iy, 256] : memref<?x?x?xf32>
        %czp_val_b2 = affine.load %czp[%iz] : memref<?xf32>
        %czm_val_b2 = affine.load %czm[%iz] : memref<?xf32>
        
        %div_cxmh_b2 = arith.divf %cxmh_val_b2, %cxph_val_b2 : f32
        %hz_term1_b2 = arith.mulf %div_cxmh_b2, %hz_old_b2 : f32
        %mui_czp_b2 = arith.mulf %mui, %czp_val_b2 : f32
        %div_czp_b2 = arith.divf %mui_czp_b2, %cxph_val_b2 : f32
        %hz_term2_b2 = arith.mulf %div_czp_b2, %tmp_val_b2 : f32
        %mui_czm_b2 = arith.mulf %mui, %czm_val_b2 : f32
        %div_czm_b2 = arith.divf %mui_czm_b2, %cxph_val_b2 : f32
        %hz_term3_b2 = arith.mulf %div_czm_b2, %bza_val_b2 : f32
        
        %sum_hz_b2 = arith.addf %hz_term1_b2, %hz_term2_b2 : f32
        %hz_new_b2 = arith.subf %sum_hz_b2, %hz_term3_b2 : f32
        affine.store %hz_new_b2, %Hz[%iz, %iy, 256] : memref<?x?x?xf32>
        
        // Bza[iz,iy,cxm] = tmp[iz,iy]
        affine.store %tmp_val_b2, %Bza[%iz, %iy, 256] : memref<?x?x?xf32>

        // Body 3: Update last y-row (y = cym) for inner x-elements
        affine.for %ix_b3 = 0 to 256 {
          // clf[iz,iy] = Ex[iz,cym,ix_b3] - Ax[iz,ix_b3] + Ey[iz,cym,ix_b3+1] - Ey[iz,cym,ix_b3]
          %ex_curr_b3 = affine.load %Ex[%iz, 256, %ix_b3] : memref<?x?x?xf32>
          %ax_val = affine.load %Ax[%iz, %ix_b3] : memref<?x?xf32>
          %ey_next_b3 = affine.load %Ey[%iz, 256, %ix_b3 + 1] : memref<?x?x?xf32>
          %ey_curr_b3 = affine.load %Ey[%iz, 256, %ix_b3] : memref<?x?x?xf32>
          
          %sub_ex_b3 = arith.subf %ex_curr_b3, %ax_val : f32
          %sub_ey_b3 = arith.subf %ey_next_b3, %ey_curr_b3 : f32
          %clf_val_b3 = arith.addf %sub_ex_b3, %sub_ey_b3 : f32
          affine.store %clf_val_b3, %clf[%iz, %iy] : memref<?x?xf32>
          
          // tmp[iz,iy] = (cymh[cym]/cyph[iy])*Bza[iz,iy,ix_b3] - (ch/cyph[iy])*clf[iz,iy]
          %cymh_val_b3 = affine.load %cymh[256] : memref<?xf32>
          %cyph_val_b3 = affine.load %cyph[%iy] : memref<?xf32>
          %bza_val_b3 = affine.load %Bza[%iz, %iy, %ix_b3] : memref<?x?x?xf32>
          
          %div_cymh_b3 = arith.divf %cymh_val_b3, %cyph_val_b3 : f32
          %term1_b3 = arith.mulf %div_cymh_b3, %bza_val_b3 : f32
          %div_ch_b3 = arith.divf %ch, %cyph_val_b3 : f32
          %term2_b3 = arith.mulf %div_ch_b3, %clf_val_b3 : f32
          %tmp_val_b3 = arith.subf %term1_b3, %term2_b3 : f32
          affine.store %tmp_val_b3, %tmp[%iz, %iy] : memref<?x?xf32>
          
          // Hz[iz,cym,ix_b3] = (cxmh[ix_b3]/cxph[ix_b3])*Hz[iz,cym,ix_b3] 
          //                 + (mui*czp[iz]/cxph[ix_b3])*tmp[iz,iy]
          //                 - (mui*czm[iz]/cxph[ix_b3])*Bza[iz,cym,ix_b3]
          %cxmh_val_b3 = affine.load %cxmh[%ix_b3] : memref<?xf32>
          %cxph_val_b3 = affine.load %cxph[%ix_b3] : memref<?xf32>
          %hz_old_b3 = affine.load %Hz[%iz, 256, %ix_b3] : memref<?x?x?xf32>
          %bza_target_b3 = affine.load %Bza[%iz, 256, %ix_b3] : memref<?x?x?xf32>
          %czp_val_b3 = affine.load %czp[%iz] : memref<?xf32>
          %czm_val_b3 = affine.load %czm[%iz] : memref<?xf32>
          
          %div_cxmh_b3 = arith.divf %cxmh_val_b3, %cxph_val_b3 : f32
          %hz_term1_b3 = arith.mulf %div_cxmh_b3, %hz_old_b3 : f32
          %mui_czp_b3 = arith.mulf %mui, %czp_val_b3 : f32
          %div_czp_b3 = arith.divf %mui_czp_b3, %cxph_val_b3 : f32
          %hz_term2_b3 = arith.mulf %div_czp_b3, %tmp_val_b3 : f32
          %mui_czm_b3 = arith.mulf %mui, %czm_val_b3 : f32
          %div_czm_b3 = arith.divf %mui_czm_b3, %cxph_val_b3 : f32
          %hz_term3_b3 = arith.mulf %div_czm_b3, %bza_target_b3 : f32
          
          %sum_hz_b3 = arith.addf %hz_term1_b3, %hz_term2_b3 : f32
          %hz_new_b3 = arith.subf %sum_hz_b3, %hz_term3_b3 : f32
          affine.store %hz_new_b3, %Hz[%iz, 256, %ix_b3] : memref<?x?x?xf32>
          
          // Bza[iz,cym,ix_b3] = tmp[iz,iy]
          affine.store %tmp_val_b3, %Bza[%iz, 256, %ix_b3] : memref<?x?x?xf32>
        }

        // Body 4: Handle last y-row (y=cym) and last x-element (ix=cxm)
        // clf[iz,iy] = Ex[iz,cym,cxm] - Ax[iz,cxm] + Ry[iz,cym] - Ey[iz,cym,cxm]
        %ex_curr_b4 = affine.load %Ex[%iz, 256, 256] : memref<?x?x?xf32>
        %ax_val_b4 = affine.load %Ax[%iz, 256] : memref<?x?xf32>
        %ry_val_b4 = affine.load %Ry[%iz, 256] : memref<?x?xf32>
        %ey_curr_b4 = affine.load %Ey[%iz, 256, 256] : memref<?x?x?xf32>
        
        %sub_ex_b4 = arith.subf %ex_curr_b4, %ax_val_b4 : f32
        %add_ry_b4 = arith.addf %sub_ex_b4, %ry_val_b4 : f32
        %clf_val_b4 = arith.subf %add_ry_b4, %ey_curr_b4 : f32
        affine.store %clf_val_b4, %clf[%iz, %iy] : memref<?x?xf32>
        
        // tmp[iz,iy] = (cymh[cym]/cyph[cym])*Bza[iz,cym,cxm] - (ch/cyph[cym])*clf[iz,iy]
        %cymh_val_b4 = affine.load %cymh[256] : memref<?xf32>
        %cyph_val_b4 = affine.load %cyph[256] : memref<?xf32>
        %bza_val_b4 = affine.load %Bza[%iz, 256, 256] : memref<?x?x?xf32>
        
        %div_cymh_b4 = arith.divf %cymh_val_b4, %cyph_val_b4 : f32
        %term1_b4 = arith.mulf %div_cymh_b4, %bza_val_b4 : f32
        %div_ch_b4 = arith.divf %ch, %cyph_val_b4 : f32
        %term2_b4 = arith.mulf %div_ch_b4, %clf_val_b4 : f32
        %tmp_val_b4 = arith.subf %term1_b4, %term2_b4 : f32
        affine.store %tmp_val_b4, %tmp[%iz, %iy] : memref<?x?xf32>
        
        // Hz[iz,cym,cxm] = (cxmh[cxm]/cxph[cxm])*Hz[iz,cym,cxm] 
        //               + (mui*czp[iz]/cxph[cxm])*tmp[iz,iy]
        //               - (mui*czm[iz]/cxph[cxm])*Bza[iz,cym,cxm]
        %cxmh_val_b4 = affine.load %cxmh[256] : memref<?xf32>
        %cxph_val_b4 = affine.load %cxph[256] : memref<?xf32>
        %hz_old_b4 = affine.load %Hz[%iz, 256, 256] : memref<?x?x?xf32>
        %czp_val_b4 = affine.load %czp[%iz] : memref<?xf32>
        %czm_val_b4 = affine.load %czm[%iz] : memref<?xf32>
        
        %div_cxmh_b4 = arith.divf %cxmh_val_b4, %cxph_val_b4 : f32
        %hz_term1_b4 = arith.mulf %div_cxmh_b4, %hz_old_b4 : f32
        %mui_czp_b4 = arith.mulf %mui, %czp_val_b4 : f32
        %div_czp_b4 = arith.divf %mui_czp_b4, %cxph_val_b4 : f32
        %hz_term2_b4 = arith.mulf %div_czp_b4, %tmp_val_b4 : f32
        %mui_czm_b4 = arith.mulf %mui, %czm_val_b4 : f32
        %div_czm_b4 = arith.divf %mui_czm_b4, %cxph_val_b4 : f32
        %hz_term3_b4 = arith.mulf %div_czm_b4, %bza_val_b4 : f32
        
        %sum_hz_b4 = arith.addf %hz_term1_b4, %hz_term2_b4 : f32
        %hz_new_b4 = arith.subf %sum_hz_b4, %hz_term3_b4 : f32
        affine.store %hz_new_b4, %Hz[%iz, 256, 256] : memref<?x?x?xf32>
        
        // Bza[iz,cym,cxm] = tmp[iz,iy]
        affine.store %tmp_val_b4, %Bza[%iz, 256, 256] : memref<?x?x?xf32>
      }
    }
    
  }
  return
  }
}
