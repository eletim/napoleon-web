import { createContext, useContext } from "react";

export interface Permissions { canPlay: boolean; isAdmin: boolean; rated?: boolean; }
// Standalone App renders and authentication-disabled local development keep existing tools.
export const PermissionsContext = createContext<Permissions>({ canPlay: true, isAdmin: true });
export const usePermissions = () => useContext(PermissionsContext);
