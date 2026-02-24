/**
 * Standard MuJoCo inverted double pendulum model XML.
 *
 * Geometry:
 *   - Cart: box (0.2 × 0.2 × 0.1 m) on a slider joint (X axis)
 *   - Pole1: capsule (0.6 m) on a hinge joint (Y axis)
 *   - Pole2: capsule (0.6 m) on a hinge joint (Y axis)
 *   - Tip site: at the top of Pole2 → z = 1.2 m when fully upright
 *
 * Action space:  1-dim motor on the slider, clipped to [-1, 1]
 * Obs space:     11-dim (see mujoco-worker.ts)
 */
export const INVERTED_DOUBLE_PENDULUM_XML = `
<mujoco model="inverted double pendulum">
  <compiler inertiafromgeom="true"/>
  <default>
    <joint armature="0.05" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true"
           exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom name="floor" pos="0 0 0" size="0 0 1" type="plane" rgba=".9 .9 .9 1"/>
    <body name="cart" pos="0 0 0">
      <geom name="cart" pos="0 0 0" quat="1 0 0 0" size=".1 .1 .05" type="box"/>
      <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0"
             range="-3 3" type="slide"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0"
               range="-100 100" type="hinge"/>
        <geom fromto="0 0 0 0 0 0.6" name="cpole"
              rgba="0 0.7 0.7 1" size=".049 .049" type="capsule"/>
        <body name="pole2" pos="0 0 .6">
          <joint axis="0 1 0" name="hinge2" pos="0 0 0"
                 range="-100 100" type="hinge"/>
          <geom fromto="0 0 0 0 0 .6" name="cpole2"
                rgba=".7 0 .7 1" size=".049 .049" type="capsule"/>
          <site name="tip" pos="0 0 .6" size=".01 .01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="100" joint="slider" name="slide"/>
  </actuator>
</mujoco>
`.trim()
