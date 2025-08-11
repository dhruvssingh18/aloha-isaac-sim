import omni.usd

stage = omni.usd.get_context().get_stage()
joint = stage.GetPrimAtPath("/World/my_joint")

# Set body0 and body1 to valid rigid body paths
joint.CreateRelationship("physics:body0", []).AddTarget("/World/object_A")
joint.CreateRelationship("physics:body1", []).AddTarget("/World/object_B")