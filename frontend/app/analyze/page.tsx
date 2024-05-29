import styles from './page.module.css';
import NavBar from "../../components/NavBar";

export default function Analyze() {
    return (
      <main>
        <NavBar/>
        <div className={styles.pageBackground}/>
      </main>
    );
  }
  