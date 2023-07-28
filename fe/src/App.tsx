import React, {useState} from 'react';
import logo from './logo.svg';
import './App.css';
import LeftNav from '../src/components/LeftNav/LeftNav';
import SatisfactionData from "./components/Pages/SatisfactionData/SatisfactionData";
import DepressionStatus from "./components/Pages/DepressionStatus/DepressionStatus";

function App() {
    const [selectedMenu, setSelectedMenu] = useState("Employee Data");
    const handleMenuChange = (menu: string) => {
        setSelectedMenu(menu);
    };
    return (
        <div className="app">
            <LeftNav selectedMenu={selectedMenu}  onChange={handleMenuChange}/>
            {selectedMenu === "Employee Data" ? (
                <div className="main-content">
                    <SatisfactionData />
                </div>
            ) : (
                <div className="main-content">
                    <DepressionStatus />
                </div>
            )}
        </div>
    );
}

export default App;
