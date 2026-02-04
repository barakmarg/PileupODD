# CLUE Algorithm Parameters Explanation

## 1. The Calculations (X and Y Axes)

### ρ (dc parameter): The Local Energy Density

- Calculated by summing the energy of all neighboring points within radius dc (30mm).
- Purpose: Quantifies how "significant" a point is. High ρ means the point is deep inside a particle shower. Low ρ means it is likely noise or a fringe hit.

### δ (dm parameter): The Distance to Nearest Higher-Density Point

- For every point, the algorithm looks for a neighbor that has a higher ρ value.
- δ is the distance to that higher-density neighbor.
- Purpose: Quantifies "isolation".
- Small δ: There is a denser point right next to me. I am likely just part of that neighbor's shower (a "Follower").
- Large δ (or Infinity): There is no denser point nearby. I am the local maximum (a "Seed").

## 2. The Cuts (The Lines)

These parameters define which points become Cluster Seeds (centers of new particles).

### rhoc (The Vertical Line)

- Rule: ρ > rhoc
- Mechanism: A hard threshold on density.
- Function: Rejects background noise. Any point to the left of this line is ignored.

### dm (The Horizontal Line)

- Rule: δ > dm
- Mechanism: A hard threshold on separation distance.
- Function: Distinguishes peaks from followers.
- If a point's nearest higher-density neighbor is further away than dm (50mm), the algorithm assumes that higher-density point belongs to a different particle. Therefore, the current point is a new Cluster Seed.

## Summary

You are tuning dc, rhoc, and dm to isolate the top-right quadrant of the plot, ensuring that every real particle generates exactly one Seed, while noise and internal shower structures are excluded.