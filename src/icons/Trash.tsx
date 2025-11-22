import { Trash2 as LucideTrash2, LucideProps } from 'lucide-react';

const Trash = ({ className, ...props }: LucideProps) => {
  return <LucideTrash2 className={className} {...props} />;
};

export default Trash;