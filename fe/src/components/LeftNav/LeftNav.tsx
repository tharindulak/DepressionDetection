import React, { useState } from "react";

//import react pro sidebar components
import {
    ProSidebar,
    Menu,
    MenuItem,
    SidebarHeader,
    SidebarFooter,
    SidebarContent,
} from "react-pro-sidebar";

//import icons from react icons
import { FaList, FaRegHeart } from "react-icons/fa";
import { FiHome, FiLogOut, FiArrowLeftCircle, FiArrowRightCircle } from "react-icons/fi";
import { RiPencilLine } from "react-icons/ri";
import { BiCog } from "react-icons/bi";


//import sidebar css from react-pro-sidebar module and our custom css
import "react-pro-sidebar/dist/css/styles.css";
import "./LeftNav.css";

const Header = (props: {selectedMenu: string, onChange: (menu: string) => void}) => {
    const {selectedMenu, onChange} = props;

    const [menuCollapse, setMenuCollapse] = useState(false);

    const changeSelectedItem = (item: string) => {
        onChange(item);
    };

    return (
        <>
            <div id="header">
                <ProSidebar collapsed={menuCollapse}>
                    <SidebarContent>
                        <Menu iconShape="square">
                            <div onClick={() => changeSelectedItem('Employee Data')}>
                                <MenuItem active={selectedMenu === "Employee Data"} icon={<FiHome/>}>
                                    Employee Data
                                </MenuItem>
                            </div>
                            <div onClick={() => changeSelectedItem('Depression Status')}>
                                <MenuItem active={selectedMenu === "Depression Status"} icon={<FaList/>}>
                                    Depression Status
                                </MenuItem>
                            </div>
                        </Menu>
                    </SidebarContent>
                    <SidebarFooter>
                        <Menu iconShape="square">
                            <MenuItem icon={<FiLogOut />}>Logout</MenuItem>
                        </Menu>
                    </SidebarFooter>
                </ProSidebar>
            </div>
        </>
    );
};

export default Header;
